import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import Papa from "papaparse";
import type { ReactNode, ElementType, PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from "react";
import {
  AreaChart, Area, ComposedChart, Line,
  BarChart, Bar, LineChart,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import {
  TrendingUp, Bot, BarChart2, Package, Warehouse,
  Send, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Search,
  Bell, Menu, LogOut,
  MapPin, RefreshCw,
  ArrowUpRight, ArrowDownRight, Building2, PackagePlus,
  Zap, FlaskConical,
} from "lucide-react";

/* ── Types ─────────────────────────────────────────────────────── */
interface AuthInfo {
  token: string;
  username: string;
  role: string;
  displayName: string;
  stores: number[];
  isAdmin: boolean;
}

/** Sản phẩm cụ thể (SKU) trả về từ /api/top-products */
interface TopProduct {
  rank: number;
  item_nbr: number;
  name?: string | null;
  family: string;
  class_code: number | null;
  perishable: number;
  unit_sales: number;
  share_pct: number;
  abc_class?: string | null;
}

/** Một dòng trong /sale_dataset_ver_2.csv (dataset lịch sử tĩnh) */
interface RawSaleData {
  date: string;
  family: string;
  unit_sales: number | null;
  perishable?: number | null;
}

/** Dòng danh mục sản phẩm thật từ /api/products */
interface ProductItem {
  item_nbr: number;
  name?: string | null;
  family: string;
  class_code: number | null;
  perishable: number;
  stock: number;
  sold_2016: number;
  status: "active" | "outofstock";
  /** Phân loại ABC theo tỷ trọng cộng dồn doanh số lịch sử (A<=80%, B<=95%, C còn lại) */
  abc_class?: string | null;
  /** Tổng dự báo 16 ngày tới của SKU (toàn chuỗi) + dải tin cậy 1σ theo RMSLE family */
  fc_total_16d?: number | null;
  fc_low?: number | null;
  fc_high?: number | null;
}

interface ProductsResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  low_stock_count: number;
  /** Ngưỡng tồn kho thấp tổng theo phạm vi (30 × số cửa hàng) - do backend tính */
  low_stock_threshold?: number;
  sales_cache_ready: boolean;
  items: ProductItem[];
}

interface Message {
  role: string;
  text: string;
}

/* ── Auth (Row-Level Isolation) ──────────────────────────────── */
const AUTH_STORAGE_KEY = "bizai_auth";

function decodeJwtPayload(token: string): Record<string, any> | null {
  try {
    const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const bin = atob(b64);
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return null;
  }
}

/** fetch tự động gắn Authorization: Bearer + header ngrok */
async function apiFetch(apiBase: string, path: string, auth: AuthInfo | null, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers || {});
  headers.set("ngrok-skip-browser-warning", "true");
  if (auth?.token) headers.set("Authorization", `Bearer ${auth.token}`);
  return fetch(`${apiBase}${path}`, { ...init, headers });
}

function restoreAuth(): AuthInfo | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const a = JSON.parse(raw) as AuthInfo;
    const payload = decodeJwtPayload(a.token);
    if (!payload?.exp || payload.exp * 1000 < Date.now()) {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      return null;
    }
    return a;
  } catch {
    return null;
  }
}

/* ── ABC badge (phân loại SKU theo tỷ trọng doanh số cộng dồn) ── */
function abcBadgeClass(c?: string | null): string {
  switch (c) {
    case "A": return "bg-blue-500/15 text-blue-600 border-blue-500/25";
    case "B": return "bg-amber-500/15 text-amber-600 border-amber-500/25";
    case "C": return "bg-slate-100 text-slate-400 border-slate-200";
    default: return "";
  }
}

/* ── Static Config ────────────────────────────────────────────── */
// Palette theo frontend2 (light theme — xanh dương chủ đạo)
const CHART_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#ef4444", "#64748b"];

const quickQuestions = [
  "Top 3 sản phẩm bán chạy nhất là gì?",
  "Doanh số ngày 2 tháng 1 năm 2016 thế nào?",
  "Tổng quan tình hình kinh doanh năm 2016?",
];

/* ── Helpers ──────────────────────────────────────────────────── */
function fmt(n: number) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + " tỷ";
  if (n >= 1e6) return (n / 1e6).toFixed(0) + " tr";
  return n.toLocaleString("vi-VN");
}

/* ── Giá tham chiếu theo nhóm hàng (ƯỚC TÍNH — dataset gốc KHÔNG có giá) ──
   Mỗi nhóm: [giá bán $/đơn vị, tỷ lệ giá vốn trên giá bán, tỷ lệ khách trả hàng]
   Công thức: Trả hàng = DT × tỷ lệ trả | LN gộp = DT − Trả hàng − Giá vốn */
const DEFAULT_ECON: [number, number, number] = [1.8, 0.78, 0.02];
const FAMILY_ECONOMICS: Record<string, [number, number, number]> = {
  "GROCERY I": [1.4, 0.8, 0.015], "GROCERY II": [1.6, 0.78, 0.02],
  "BEVERAGES": [1.2, 0.74, 0.01], "CLEANING": [2.3, 0.72, 0.015],
  "PRODUCE": [1.0, 0.82, 0.04], "DAIRY": [1.5, 0.8, 0.03],
  "BREAD/BAKERY": [0.9, 0.7, 0.03], "POULTRY": [3.4, 0.84, 0.02],
  "MEATS": [4.2, 0.84, 0.02], "DELI": [3.6, 0.8, 0.02],
  "EGGS": [2.0, 0.82, 0.03], "FROZEN FOODS": [2.6, 0.76, 0.02],
  "PREPARED FOODS": [2.3, 0.72, 0.03], "SEAFOOD": [5.0, 0.85, 0.04],
  "LIQUOR,WINE,BEER": [4.5, 0.72, 0.005], "PERSONAL CARE": [2.6, 0.68, 0.02],
  "HOME CARE": [3.1, 0.7, 0.015], "BEAUTY": [3.4, 0.65, 0.03],
  "HOME AND KITCHEN I": [6.5, 0.68, 0.03], "HOME AND KITCHEN II": [5.5, 0.68, 0.03],
  "HOME APPLIANCES": [28, 0.8, 0.04], "PLAYERS AND ELECTRONICS": [35, 0.84, 0.05],
  "HARDWARE": [4.8, 0.72, 0.02], "PET SUPPLIES": [3.2, 0.72, 0.02],
  "SCHOOL AND OFFICE SUPPLIES": [2.4, 0.66, 0.02], "BABY CARE": [3.3, 0.72, 0.03],
  "CELEBRATION": [2.8, 0.62, 0.04], "LADIESWEAR": [7.5, 0.6, 0.06],
  "LINGERIE": [5.5, 0.58, 0.06], "LAWN AND GARDEN": [6, 0.66, 0.03],
  "BOOKS": [4.2, 0.62, 0.02], "MAGAZINES": [1.8, 0.55, 0.02],
  "AUTOMOTIVE": [9.5, 0.74, 0.02], "ABRASIVES": [2.1, 0.68, 0.02],
};
function econOf(fam: string): [number, number, number] {
  const key = (fam || "").toUpperCase().trim();
  return FAMILY_ECONOMICS[key] ?? DEFAULT_ECON;
}
/* Tỷ giá quy đổi HIỂN THỊ: dữ liệu gốc trong DB tính bằng USD,
   giao diện hiển thị VND. Đổi tỷ giá chỉ cần sửa con số này. */
const USD_TO_VND = 25500;
function fmtMoney(n: number): string {
  const vnd = Math.round(n * USD_TO_VND);
  return vnd.toLocaleString("vi-VN") + " ₫";
}
function fmtMoneyCompact(v: number): string {
  const x = v * USD_TO_VND;
  if (Math.abs(x) >= 1e12) return `${(x / 1e12).toLocaleString("vi-VN", { maximumFractionDigits: 2 })} nghìn tỷ ₫`;
  if (Math.abs(x) >= 1e9) return `${(x / 1e9).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} tỷ ₫`;
  if (Math.abs(x) >= 1e6) return `${Math.round(x / 1e6).toLocaleString("vi-VN")} tr ₫`;
  return `${Math.round(x).toLocaleString("vi-VN")} ₫`;
}

/* Cache module-level: tránh tải lại CSV / gọi lại API mỗi lần chuyển tab */
let csvHistCache: RawSaleData[] | null = null;
type BizPoint = { date: string; revenue: number; returns: number; gross_profit: number; cogs?: number; invoices?: number };
const bizCacheByBranch: Record<string, BizPoint[]> = {};

/* ── Small components (kiểu frontend2) ───────────────────────── */
function StatCard({
  label, value, sub, trend, color = "blue",
}: { label: string; value: string; sub: string; trend: "up" | "down"; color?: string }) {
  const gradients: Record<string, string> = {
    blue: "from-blue-500/10 to-transparent border-blue-500/20",
    green: "from-emerald-500/10 to-transparent border-emerald-500/20",
    amber: "from-amber-500/10 to-transparent border-amber-500/20",
    purple: "from-purple-500/10 to-transparent border-purple-500/20",
  };
  return (
    <div className={`bg-gradient-to-br ${gradients[color]} border rounded-xl p-5 bg-white`}>
      <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-2">{label}</p>
      <p className="text-2xl font-semibold text-slate-900 tabular-nums leading-none mb-2">{value}</p>
      <div className="flex items-center gap-1">
        {trend === "up"
          ? <ArrowUpRight size={13} className="text-emerald-500 flex-shrink-0" />
          : <ArrowDownRight size={13} className="text-red-500 flex-shrink-0" />}
        <span className={`text-xs font-mono ${trend === "up" ? "text-emerald-500" : "text-red-500"}`}>{sub}</span>
      </div>
    </div>
  );
}

function SectionHeader({
  title, sub, action,
}: { title: string; sub?: string; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h1 className="text-lg font-semibold text-slate-900 tracking-tight">{title}</h1>
        {sub && <p className="text-xs font-mono text-slate-500 mt-0.5">{sub}</p>}
      </div>
      {action}
    </div>
  );
}

function ChartTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-2xl">
      <p className="text-[10px] font-mono text-slate-500 mb-1.5">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color }} className="text-xs font-mono">
          {p.name}: <span className="font-medium">{typeof p.value === "number" ? p.value.toLocaleString("vi-VN") : p.value}</span>
        </p>
      ))}
    </div>
  );
}

/* Tooltip cho các biểu đồ tiền tệ (dữ liệu gốc USD, hiển thị VND) */
function BizTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-2xl">
      <p className="text-[10px] font-mono text-slate-500 mb-1.5">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color }} className="text-xs font-mono">
          {p.name}: <span className="font-medium">{fmtMoney(Number(p.value))}</span>
        </p>
      ))}
    </div>
  );
}

/* ── Section: Đăng nhập ───────────────────────────────────────── */
function LoginView({ apiBase, setApiBase, onLoggedIn }: {
  apiBase: string;
  setApiBase: (v: string) => void;
  onLoggedIn: (a: AuthInfo) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    setErr("");
    try {
      const res = await fetch(`${apiBase}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `Lỗi ${res.status}`);
      onLoggedIn({
        token: data.access_token,
        username: data.username,
        role: data.role,
        displayName: data.display_name || data.username,
        stores: Array.isArray(data.stores) ? data.stores : [],
        isAdmin: data.role === "admin",
      });
    } catch (e: any) {
      setErr(e?.message || "Không thể đăng nhập. Kiểm tra backend đã chạy chưa.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-slate-50" style={{ fontFamily: "'Be Vietnam Pro', system-ui, sans-serif" }}>
      <div className="w-full max-w-sm bg-white border border-slate-200 rounded-2xl p-7 shadow-sm">
        <div className="flex items-center gap-2.5 mb-6">
          <div className="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center">
            <Warehouse size={17} className="text-white" />
          </div>
          <div>
            <p className="text-base font-semibold text-slate-900 leading-none">BizAI</p>
            <p className="text-[10px] font-mono text-slate-500 mt-0.5">Retail Intelligence</p>
          </div>
        </div>

        <label className="block text-[10px] font-mono text-slate-500 uppercase tracking-wider mb-1">Tên đăng nhập</label>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          autoFocus
          className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-900 mb-3 focus:outline-none focus:border-blue-500/50"
          placeholder="vd: manager1"
        />

        <label className="block text-[10px] font-mono text-slate-500 uppercase tracking-wider mb-1">Mật khẩu</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-900 focus:outline-none focus:border-blue-500/50"
          placeholder="••••••••"
        />

        {err && (
          <div className="mt-3 bg-red-500/10 border border-red-500/25 rounded-lg px-3 py-2 text-[11px] text-red-600">{err}</div>
        )}

        <button
          onClick={submit}
          disabled={busy || !username.trim() || !password}
          className="w-full mt-4 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white text-xs font-medium py-2.5 rounded-lg transition-colors"
        >
          {busy ? "Đang đăng nhập..." : "Đăng nhập"}
        </button>

        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-wider mt-5 mb-1">Backend API</label>
        <input
          value={apiBase}
          onChange={(e) => setApiBase(e.target.value)}
          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-[10px] font-mono text-slate-600 focus:outline-none focus:border-blue-500/40"
          placeholder="http://localhost:8000"
        />
      </div>
    </div>
  );
}

/* ── Section: Dự báo doanh số ─────────────────────────────────── */
function SalesForecast({ branchId, apiBase, auth, onAuthError }: {
  branchId: string; apiBase: string; auth: AuthInfo; onAuthError: () => void;
}) {
  const [liveChart, setLiveChart] = useState<Array<{ date: string; dubao: number }> | null>(null);
  const [liveKpi, setLiveKpi] = useState<{ total: string; avgPerDay: string; days: number | null } | null>(null);
  const [topProducts, setTopProducts] = useState<TopProduct[]>([]);
  const [topErr, setTopErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [apiOk, setApiOk] = useState(false);
  const [apiErr, setApiErr] = useState("");

  // Bộ lọc ngành hàng + mốc thời gian
  const [family, setFamily] = useState("all");
  const [rangeMode, setRangeMode] = useState<"all" | 7 | 14 | 30 | "custom">("all");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [families, setFamilies] = useState<string[]>([]);
  const [meta, setMeta] = useState<{ from: string; to: string } | null>(null);

  const isRealStore = branchId === "all" || /^\d+$/.test(branchId);

  // Danh sách ngành hàng cho dropdown (lấy một lần khi mở view)
  useEffect(() => {
    let cancelled = false;
    apiFetch(apiBase, "/api/product-families", auth)
      .then((res) => {
        if (res.status === 401) { onAuthError(); return Promise.reject(); }
        return res.ok ? res.json() : Promise.reject();
      })
      .then((list) => { if (!cancelled && Array.isArray(list)) setFamilies(list); })
      .catch(() => { /* dropdown rỗng vẫn xem được tổng hợp */ });
    return () => { cancelled = true; };
  }, [apiBase, auth, onAuthError]);

  // Biên thời gian dữ liệu dự báo theo cửa hàng đang chọn
  useEffect(() => {
    if (!isRealStore) { setMeta(null); return; }
    let cancelled = false;
    const storeParam = branchId !== "all" ? `?store_nbr=${branchId}` : "";
    apiFetch(apiBase, `/api/forecast-meta${storeParam}`, auth)
      .then((res) => {
        if (res.status === 401) { onAuthError(); return Promise.reject(); }
        return res.ok ? res.json() : Promise.reject();
      })
      .then((m) => { if (!cancelled && m?.date_from && m?.date_to) setMeta({ from: m.date_from, to: m.date_to }); })
      .catch(() => { if (!cancelled) setMeta(null); });
    return () => { cancelled = true; };
  }, [branchId, apiBase, auth, isRealStore, onAuthError]);

  // Tính khoảng ngày hiệu lực theo chip mốc thời gian đang chọn
  useEffect(() => {
    if (!meta) return;
    if (rangeMode === "all") { setFromDate(""); setToDate(""); return; }
    if (rangeMode === "custom") return; // người dùng tự nhập
    const startMs = Date.parse(meta.from + "T00:00:00Z");
    const endBoundMs = Date.parse(meta.to + "T00:00:00Z");
    if (Number.isNaN(startMs)) return;
    const endMs = Math.min(startMs + ((rangeMode as number) - 1) * 86400000, Number.isNaN(endBoundMs) ? Infinity : endBoundMs);
    setFromDate(meta.from);
    setToDate(new Date(endMs).toISOString().slice(0, 10));
  }, [rangeMode, meta]);

  // Top sản phẩm bán chạy (SKU cụ thể) theo phạm vi cửa hàng đang chọn
  useEffect(() => {
    if (!isRealStore) return;
    let cancelled = false;
    setTopErr("");
    const storeParam = branchId !== "all" ? `&store_nbr=${branchId}` : "";
    apiFetch(apiBase, `/api/top-products?limit=6${storeParam}`, auth)
      .then((res) => {
        if (res.status === 401) { onAuthError(); return Promise.reject(); }
        return res.ok ? res.json() : res.json().then((d) => { throw new Error(d?.detail || `Lỗi ${res.status}`); });
      })
      .then((data) => { if (!cancelled) setTopProducts(data.items || []); })
      .catch((e) => { if (!cancelled) { setTopProducts([]); setTopErr(e?.message || "Không tải được top sản phẩm."); } });
    return () => { cancelled = true; };
  }, [branchId, apiBase, auth, isRealStore, onAuthError]);

  useEffect(() => {
    if (!isRealStore) {
      setApiOk(false);
      setLoading(false);
      return;
    }

    let cancelled = false;
    async function load() {
      setLoading(true);
      setApiErr("");
      try {
        const q = new URLSearchParams();
        if (branchId !== "all") q.set("store_nbr", branchId);
        if (family !== "all") q.set("family", family);
        if (fromDate) q.set("date_from", fromDate);
        if (toDate) q.set("date_to", toDate);
        const qs = q.toString();
        const suffix = qs ? `?${qs}` : "";
        const [predsRes, kpiRes] = await Promise.all([
          apiFetch(apiBase, `/api/predictions${suffix}`, auth),
          apiFetch(apiBase, `/api/kpi${suffix}`, auth),
        ]);
        if (predsRes.status === 401 || kpiRes.status === 401) { onAuthError(); return; }
        if (!predsRes.ok || !kpiRes.ok) {
          const detail = await (predsRes.status !== 200 ? predsRes : kpiRes).json().catch(() => null);
          throw new Error(typeof detail?.detail === "string" ? detail.detail : "API lỗi");
        }
        const preds = await predsRes.json();
        const kpiData = await kpiRes.json();
        if (cancelled) return;

        const byDate: Record<string, number> = {};
        for (const p of preds) {
          byDate[p.date] = (byDate[p.date] ?? 0) + p.predicted_sales;
        }
        const chart = Object.entries(byDate)
          .sort(([a], [b]) => (a < b ? -1 : 1))
          .map(([date, dubao]) => ({ date: date.slice(5), dubao: Math.round(dubao) }));

        setLiveChart(chart);
        setLiveKpi({
          total: fmt(kpiData.total_predicted_sales),
          avgPerDay: fmt(kpiData.avg_per_day),
          days: typeof kpiData.forecast_days === "number" ? kpiData.forecast_days : null,
        });
        setApiOk(true);
      } catch (e: any) {
        if (!cancelled) {
          setApiErr(e?.message || "Không kết nối được backend.");
          setApiOk(false);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [branchId, apiBase, auth, isRealStore, onAuthError, family, fromDate, toDate]);

  const chartData = liveChart ?? [];

  if (!isRealStore || loading) {
    return (
      <div className="space-y-5">
        <SectionHeader title="Dự báo doanh số" sub="Phân tích & dự báo doanh thu theo thời gian thực" />
        <div className="flex items-center justify-center h-64 text-slate-500 text-sm gap-2 bg-white border border-slate-200 rounded-xl">
          <RefreshCw size={15} className="animate-spin" />
          Đang tải dữ liệu từ backend...
        </div>
      </div>
    );
  }

  if (!apiOk) {
    return (
      <div className="space-y-5">
        <SectionHeader title="Dự báo doanh số" sub="Phân tích & dự báo doanh thu theo thời gian thực" />
        <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-3 text-sm text-red-600">
          {apiErr || `Không kết nối được backend (${apiBase}). Kiểm tra lại backend đã chạy chưa.`}
        </div>
      </div>
    );
  }

  const familyLabel = family === "all" ? "Tất cả ngành hàng" : family;
  const rangeLabel = rangeMode !== "all" && fromDate && toDate
    ? `${formatDateVN(fromDate)} → ${formatDateVN(toDate)}`
    : "Toàn kỳ dự báo";

  return (
    <div className="space-y-5">
      <SectionHeader title="Dự báo doanh số" sub="Phân tích & dự báo doanh thu theo thời gian thực — Nguồn: model LightGBM thật" />

      {/* Bộ lọc ngành hàng + mốc thời gian */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center gap-x-4 gap-y-3">
        <label className="flex items-center gap-2 text-xs text-slate-500">
          <Package size={12} className="text-slate-400 shrink-0" />
          Ngành hàng
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value)}
            className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:border-blue-500/40 max-w-[220px]"
          >
            <option value="all">Tất cả ngành hàng</option>
            {families.map((f) => (<option key={f} value={f}>{f}</option>))}
          </select>
        </label>
        <div className="flex gap-1.5 flex-wrap">
          {([["all", "Toàn kỳ"], [7, "7 ngày"], [14, "14 ngày"], [30, "30 ngày"], ["custom", "Tùy chỉnh"]] as [string | number, string][]).map(([mode, label]) => (
            <button
              key={String(mode)}
              onClick={() => setRangeMode(mode as typeof rangeMode)}
              className={`text-xs px-3 py-2 rounded-lg border transition-colors ${rangeMode === mode ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-white border-slate-200 text-slate-500 hover:text-slate-900"}`}
            >
              {label}
            </button>
          ))}
        </div>
        {rangeMode === "custom" && (
          <>
            <label className="flex items-center gap-1.5 text-xs text-slate-500">
              Từ ngày
              <input type="date" value={fromDate} min={meta?.from || undefined} max={meta?.to || undefined}
                onChange={(e) => setFromDate(e.target.value)} disabled={!meta}
                className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-500/40 disabled:opacity-40" />
            </label>
            <label className="flex items-center gap-1.5 text-xs text-slate-500">
              Đến ngày
              <input type="date" value={toDate} min={meta?.from || undefined} max={meta?.to || undefined}
                onChange={(e) => setToDate(e.target.value)} disabled={!meta}
                className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-500/40 disabled:opacity-40" />
            </label>
            {(fromDate || toDate) && (
              <button
                onClick={() => { setFromDate(""); setToDate(""); }}
                className="text-xs text-slate-400 hover:text-blue-600 underline underline-offset-2 transition-colors"
              >
                Xóa khoảng
              </button>
            )}
          </>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {liveKpi && (
          <>
            <StatCard label={`Tổng dự báo (${rangeLabel})`} value={liveKpi.total}
              sub={`Ngành hàng: ${familyLabel}`} trend="up" color="blue" />
            <StatCard label="Trung bình mỗi ngày" value={liveKpi.avgPerDay}
              sub={`Theo bộ lọc hiện tại · ${liveKpi.days ?? "..."} ngày`} trend="up" color="green" />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 bg-white border border-slate-200 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-900">Dự báo doanh số theo ngày</p>
              <p className="text-[10px] font-mono text-slate-500 mt-0.5">{familyLabel} · {rangeLabel}</p>
            </div>
            <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-purple-500 rounded inline-block" />Dự báo</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="gForecast" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a855f7" stopOpacity={0.18} />
                  <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false}
                tickFormatter={(v) => v >= 1000 ? `${v / 1000}B` : `${v}M`} />
              <Tooltip content={<ChartTip />} />
              <Area type="monotone" dataKey="dubao" name="Dự báo" stroke="none" fill="url(#gForecast)" />
              <Line type="monotone" dataKey="dubao" name="Dự báo" stroke="#a855f7" strokeWidth={2}
                dot={{ fill: "#a855f7", r: 3, strokeWidth: 0 }} activeDot={{ r: 4 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <div className="flex items-baseline justify-between mb-4">
            <p className="text-sm font-semibold text-slate-900">Top sản phẩm bán chạy</p>
            <p className="text-[9px] font-mono text-slate-400 uppercase">Doanh số 2016</p>
          </div>
          <div className="space-y-3">
            {topErr ? (
              <p className="text-[11px] text-red-500 bg-red-500/10 border border-red-500/20 rounded-lg px-2.5 py-2">{topErr}</p>
            ) : topProducts.length === 0 ? (
              <p className="text-slate-500 text-xs flex items-center gap-1.5"><RefreshCw size={11} className="animate-spin" /> Đang tải dữ liệu...</p>
            ) : (
              topProducts.map((p, i) => (
                <div key={p.item_nbr}>
                  <div className="flex items-center justify-between mb-1 gap-2">
                    <span className="text-slate-700 text-xs truncate flex items-center gap-1.5 min-w-0">
                      <span className={`font-mono text-[9px] w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0 ${i === 0 ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-500"}`}>{p.rank}</span>
                      <span className="font-medium truncate" title={p.name ?? undefined}>{p.name ?? `#${p.item_nbr}`}</span>
                      <span className="text-[10px] font-mono text-slate-400 flex-shrink-0">#{p.item_nbr}</span>
                      <span className="text-[10px] font-mono text-slate-400 truncate">{p.family}</span>
                      {p.abc_class && (
                        <span className={`text-[9px] font-mono px-1 rounded border flex-shrink-0 ${abcBadgeClass(p.abc_class)}`} title={`Lớp ${p.abc_class} theo tỷ trọng doanh số cộng dồn`}>
                          {p.abc_class}
                        </span>
                      )}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400 flex-shrink-0">{p.share_pct}%</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all duration-700"
                      style={{ width: `${(p.unit_sales / (topProducts[0]?.unit_sales || 1)) * 100}%`, background: CHART_COLORS[i % CHART_COLORS.length] }}
                    />
                  </div>
                  <p className="text-[10px] font-mono text-slate-500 mt-0.5">{Math.round(p.unit_sales).toLocaleString("vi-VN")} đơn vị{p.perishable === 1 ? " · dễ hỏng" : ""}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Section: AI Chatbot (Gọi qua Backend Gateway -> Groq Qwen 3) ─────────── */
function AIChatbot({ apiBase, auth, onAuthError }: { apiBase: string; auth: AuthInfo; onAuthError: () => void }) {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Xin chào! Tôi là AI Assistant quản trị chuỗi cung ứng (Groq Qwen 3.6). Hãy hỏi về tồn kho, dự báo nhu cầu hoặc kế hoạch đặt hàng nhé!" }
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Lịch sử hội thoại gửi kèm backend để AI nhớ ngữ cảnh (định dạng OpenAI messages)
  const historyRef = useRef<{ role: "user" | "assistant"; content: string }[]>([]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  const send = async (text: string) => {
    if (!text.trim() || thinking) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setThinking(true);

    try {
      // Gọi Backend Gateway (/api/chat) -> LLM Agent Groq Qwen 3.6 + Function Calling
      // Gửi kèm tối đa 40 tin gần nhất; backend tự trim theo ngân sách token nên
      // không cần cắt cứng ở đây (tránh mất ngữ cảnh gần khi câu trả lời dài).
      const response = await apiFetch(apiBase, "/api/chat", auth, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_query: text, chat_history: historyRef.current.slice(-40) })
      });

      if (response.status === 401) { onAuthError(); return; }
      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        const detail = typeof errData?.detail === "string" ? errData.detail : await response.text();
        console.error("Backend /api/chat lỗi chi tiết:", detail);
        throw new Error(typeof detail === "string" && detail.length < 300 ? detail : `Chi tiết lỗi từ server: ${detail}`);
      }

      const data = await response.json();
      const aiText = data?.reply || "Tôi không hiểu phản hồi từ AI.";
      // Chỉ ghi cặp hỏi-đáp vào lịch sử khi thành công (lỗi không nhiễm ngữ cảnh)
      historyRef.current = [
        ...historyRef.current,
        { role: "user", content: text },
        { role: "assistant", content: aiText },
      ];
      setMessages((m) => [...m, { role: "assistant", text: aiText }]);
    } catch (error: any) {
      console.error("Lỗi hệ thống:", error);
      setMessages((m) => [...m, { role: "assistant", text: `Lỗi AI: ${error.message}` }]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <div className="flex flex-col xl:flex-row gap-4 h-[calc(100vh-10rem)] min-h-[480px]">
      <div className="flex-1 min-w-0 flex flex-col bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-blue-50 flex items-center justify-center">
            <Bot size={14} className="text-blue-600" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">AI Business Assistant</p>
            <p className="text-[10px] font-mono text-slate-500">Powered by Groq Qwen 3.6</p>
          </div>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
            <span className="text-[10px] font-mono text-emerald-500">Online</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-2.5 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${m.role === "assistant" ? "bg-blue-50" : "bg-slate-100"}`}>
                {m.role === "assistant" ? <Bot size={13} className="text-blue-600" /> : <span className="text-[10px] text-slate-500 font-mono font-bold">N</span>}
              </div>
              <div className={`flex flex-col gap-1 max-w-[80%] ${m.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`rounded-2xl px-4 py-2.5 text-xs leading-relaxed ${m.role === "assistant" ? "bg-slate-100 text-slate-900 rounded-tl-sm" : "bg-blue-600 text-white rounded-tr-sm"}`}>
                  {m.text.split("\n").map((line, j) => (
                    <p key={j} className={line === "" ? "h-2" : ""}>
                      {line.startsWith("**") && line.endsWith("**")
                        ? <strong className={m.role === "assistant" ? "text-blue-600" : "text-white"}>{line.slice(2, -2)}</strong>
                        : line}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          ))}
          {thinking && (
            <div className="flex gap-2.5">
              <div className="w-7 h-7 rounded-full bg-blue-50 flex items-center justify-center">
                <Bot size={13} className="text-blue-600" />
              </div>
              <div className="bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-slate-200 p-3">
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send(input)}
              placeholder="Hỏi AI về dữ liệu cửa hàng..."
              className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500/40 transition-colors"
            />
            <button
              onClick={() => send(input)}
              disabled={!input.trim() || thinking}
              className="bg-blue-600 text-white px-3.5 rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>

      <div className="xl:w-52 shrink-0 flex flex-col gap-3 overflow-y-auto scrollbar-hide">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs font-semibold text-slate-900 mb-3">Câu hỏi nhanh</p>
          <div className="space-y-1.5">
            {quickQuestions.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                className="w-full text-left text-[11px] text-slate-600 hover:text-blue-600 bg-slate-50 hover:bg-blue-50 border border-transparent hover:border-blue-200 rounded-lg px-3 py-2 transition-all leading-snug"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs font-semibold text-slate-900 mb-3">Trạng thái</p>
          <div className="space-y-2.5">
            {[{ label: "Model", val: "Groq Qwen 3.6" }, { label: "Gateway", val: apiBase.replace(/^https?:\/\//, "") }].map((s) => (
              <div key={s.label} className="flex items-center justify-between gap-2">
                <span className="text-slate-500 text-xs">{s.label}</span>
                <span className="text-slate-900 font-mono text-[10px] font-medium truncate">{s.val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Section: Phân tích sản phẩm ─────────────────────────────── */
type HistGranularity = "day" | "month" | "quarter";

function histPeriodKey(dateStr: string, g: HistGranularity): string {
  if (g === "day") return dateStr;
  if (g === "month") return dateStr.slice(0, 7);
  const q = Math.floor((parseInt(dateStr.slice(5, 7), 10) - 1) / 3) + 1;
  return `${dateStr.slice(0, 4)}-Q${q}`;
}

function histPeriodLabel(key: string, g: HistGranularity): string {
  if (g === "day") { const [, m, d] = key.split("-"); return `${d}/${m}`; }
  if (g === "month") { const [y, m] = key.split("-"); return `${m}/${y.slice(2)}`; }
  const [y, q] = key.split("-Q");
  return `Q${q}/${y}`;
}

function formatDateVN(s: string): string {
  return s ? s.split("-").reverse().join("/") : "—";
}

function InsightCard({ tone, children }: { tone: "good" | "warn" | "bad" | "info"; children: ReactNode }) {
  const tones: Record<string, string> = {
    good: "bg-emerald-500/5 border-emerald-500/15",
    warn: "bg-amber-500/5 border-amber-500/15",
    bad: "bg-red-500/5 border-red-500/15",
    info: "bg-blue-500/5 border-blue-500/15",
  };
  const bars: Record<string, string> = { good: "bg-emerald-500", warn: "bg-amber-500", bad: "bg-red-500", info: "bg-blue-500" };
  return (
    <div className={`flex gap-2.5 p-3 rounded-xl border text-xs leading-relaxed text-slate-700 ${tones[tone]}`}>
      <div className={`w-1 flex-shrink-0 rounded-full mt-0.5 ${bars[tone]}`} style={{ minHeight: "1rem" }} />
      <div>{children}</div>
    </div>
  );
}

function ProductAnalysis({ branchId, apiBase, auth, onAuthError }: {
  branchId: string; apiBase: string; auth: AuthInfo; onAuthError: () => void;
}) {
  // ── Dữ liệu dự báo từ backend (retail.db) ──
  const [trendRows, setTrendRows] = useState<Array<Record<string, any>>>([]);
  const [families, setFamilies] = useState<string[]>([]);
  const [horizon, setHorizon] = useState<{ from: string; to: string } | null>(null);
  const [trendDays, setTrendDays] = useState(16);
  const [totalForecast, setTotalForecast] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setErr("");
      try {
        const storeParam = branchId !== "all" ? `&store_nbr=${branchId}` : "";
        const trendRes = await apiFetch(apiBase, `/api/family-trend?days=${trendDays}&top_families=6${storeParam}`, auth);
        if (trendRes.status === 401) { onAuthError(); return; }
        if (!trendRes.ok) {
          const d = await trendRes.json().catch(() => null);
          throw new Error(typeof d?.detail === "string" ? d.detail : "Không tải được xu hướng dự báo.");
        }
        const trend = await trendRes.json();
        if (cancelled) return;

        setTrendRows(trend.series || []);
        setFamilies(trend.families || []);
        setHorizon({ from: trend.date_from, to: trend.date_to });
        setTotalForecast((trend.series || []).reduce((s: number, r: any) => {
          return s + (trend.families || []).reduce((s2: number, f: string) => s2 + (Number(r[f]) || 0), 0);
        }, 0));
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Không kết nối được backend.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [branchId, apiBase, auth, onAuthError, trendDays]);

  // Ngày cuối cùng thực tế của cửa sổ đang chọn (backend có thể trả ít điểm hơn kỳ gốc)
  const winLastDate: string = trendRows.length ? String((trendRows[trendRows.length - 1] as any).date || "") : "";

  // ── Dữ liệu bán hàng LỊCH SỬ (dataset tổng toàn chuỗi trong public/) ──
  const [histRows, setHistRows] = useState<RawSaleData[]>([]);
  useEffect(() => {
    if (csvHistCache) { setHistRows(csvHistCache); return; }
    let cancelled = false;
    Papa.parse<RawSaleData>("/sale_dataset_ver_2.csv", {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (result) => {
        const rows = result.data.filter((d) => d.date && d.family && d.unit_sales != null);
        csvHistCache = rows;
        if (!cancelled) setHistRows(rows);
      },
      error: () => { /* CSV thiếu -> các mục lịch sử hiện trạng thái trống */ },
    });
    return () => { cancelled = true; };
  }, []);

  // ── Chỉ số kinh doanh lịch sử từ backend (RLS theo cửa hàng; gốc USD, hiển thị VND) ──
  const [bizRows, setBizRows] = useState<BizPoint[]>([]);
  const [bizState, setBizState] = useState<"loading" | "ok" | "none">("loading");
  useEffect(() => {
    const cached = bizCacheByBranch[branchId];
    if (cached) { setBizRows(cached); setBizState("ok"); return; }
    setBizRows([]);
    setBizState("loading");
    let cancelled = false;
    apiFetch(apiBase, `/api/metrics/history${branchId !== "all" ? `?store_nbr=${branchId}` : ""}`, auth)
      .then(async (r) => {
        if (cancelled) return;
        if (r.status === 401) { onAuthError(); return; }
        if (!r.ok) { setBizState("none"); return; }
        const d = await r.json();
        const items: BizPoint[] = d.items || [];
        bizCacheByBranch[branchId] = items;
        if (!cancelled) {
          setBizRows(items);
          setBizState(items.length ? "ok" : "none");
        }
      })
      .catch(() => { if (!cancelled) setBizState("none"); });
    return () => { cancelled = true; };
  }, [branchId, apiBase, auth, onAuthError]);

  // ── Bộ lọc thời gian phần lịch sử ──
  const [granularity, setGranularity] = useState<HistGranularity>("month");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [specificDate, setSpecificDate] = useState("");

  // Khung thời gian: ưu tiên CSV; nếu CSV thiếu thì dùng chính dữ liệu backend
  const bounds = useMemo(() => {
    let min = ""; let max = "";
    for (const r of histRows) {
      if (!min || r.date < min) min = r.date;
      if (!max || r.date > max) max = r.date;
    }
    if (!min) {
      for (const p of bizRows) {
        if (!min || p.date < min) min = p.date;
        if (!max || p.date > max) max = p.date;
      }
    }
    return { min, max };
  }, [histRows, bizRows]);

  const range = useMemo(() => {
    let from = fromDate || bounds.min;
    let to = toDate || bounds.max;
    if (from && to && from > to) [from, to] = [to, from];
    return { from, to };
  }, [fromDate, toDate, bounds]);

  // Chọn "Ngày cụ thể" -> toàn bộ biểu đồ/KPI co lại đúng NGÀY đó
  // (ghi đè khoảng từ-đến và thang hiển thị cho đến khi bấm Xóa)
  const effRange = useMemo(
    () => (specificDate ? { from: specificDate, to: specificDate } : range),
    [specificDate, range.from, range.to]
  );
  // Độ dài kỳ hiệu lực (số ngày)
  const customSpanDays = useMemo(() => {
    if (specificDate) return 1;
    if (!effRange.from || !effRange.to) return Number.POSITIVE_INFINITY;
    const d = Math.round((Date.parse(effRange.to) - Date.parse(effRange.from)) / 86400000) + 1;
    return Number.isFinite(d) && d > 0 ? d : Number.POSITIVE_INFINITY;
  }, [specificDate, effRange.from, effRange.to]);
  // Kỳ ngắn (<=61 ngày) tự dùng thang NGÀY để không bị gộp thành cột tháng/quý gây hiểu nhầm
  const effGranularity: HistGranularity =
    specificDate || customSpanDays <= 61 ? "day" : granularity;

  const dailyTotals = useMemo(() => {
    const m: Record<string, number> = {};
    for (const r of histRows) m[r.date] = (m[r.date] || 0) + (Number(r.unit_sales) || 0);
    return m;
  }, [histRows]);
  const allDates = useMemo(() => Object.keys(dailyTotals).sort(), [dailyTotals]);

  const agg = useMemo(() => {
    const map = new Map<string, { total: number; perishable: number; durable: number; fams: Record<string, number> }>();
    for (const r of histRows) {
      if (r.date < effRange.from || r.date > effRange.to) continue;
      const key = histPeriodKey(r.date, effGranularity);
      let bucket = map.get(key);
      if (!bucket) { bucket = { total: 0, perishable: 0, durable: 0, fams: {} }; map.set(key, bucket); }
      const v = Number(r.unit_sales) || 0;
      bucket.total += v;
      if (r.perishable === 1) bucket.perishable += v; else bucket.durable += v;
      bucket.fams[r.family] = (bucket.fams[r.family] || 0) + v;
    }
    const series = Array.from(map.entries())
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([period, b]) => ({ period, label: histPeriodLabel(period, effGranularity), ...b, growth: null as number | null }));
    for (let i = 1; i < series.length; i++) {
      const prev = series[i - 1].total;
      series[i].growth = prev > 0 ? ((series[i].total - prev) / prev) * 100 : null;
    }
    const famTotals: Record<string, number> = {};
    for (const s of series) for (const [fam, v] of Object.entries(s.fams)) famTotals[fam] = (famTotals[fam] || 0) + v;
    const topFamilies = Object.entries(famTotals).sort(([, a], [, b]) => b - a);
    return { series, topFamilies };
  }, [histRows, effRange.from, effRange.to, effGranularity]);

  // Chỉ số kinh doanh: ưu tiên backend /api/metrics/history (RLS, nhanh);
  // fallback quy đổi từ CSV theo giá tham chiếu nếu route chưa khả dụng.
  const trendChartData = useMemo(() => {
    if (bizRows.length) {
      const m = new Map<string, { label: string; revenue: number; returns: number; grossProfit: number }>();
      for (const p of bizRows) {
        if (p.date < effRange.from || p.date > effRange.to) continue;
        const key = histPeriodKey(p.date, effGranularity);
        let e = m.get(key);
        if (!e) { e = { label: histPeriodLabel(key, effGranularity), revenue: 0, returns: 0, grossProfit: 0 }; m.set(key, e); }
        e.revenue += p.revenue; e.returns += p.returns; e.grossProfit += p.gross_profit;
      }
      return Array.from(m.values()).map((e) => ({
        label: e.label,
        revenue: Math.round(e.revenue),
        returns: Math.round(e.returns),
        grossProfit: Math.round(e.grossProfit),
      }));
    }
    return agg.series.map((s) => {
      let revenue = 0, cogs = 0, returns = 0;
      for (const [fam, qty] of Object.entries(s.fams)) {
        const [price, costRatio, returnRate] = econOf(fam);
        const rev = qty * price;
        revenue += rev;
        returns += rev * returnRate;
        cogs += qty * (1 - returnRate) * price * costRatio; // giá vốn tính trên số bán ròng
      }
      return {
        label: s.label,
        revenue: Math.round(revenue),
        returns: Math.round(returns),
        grossProfit: Math.round(revenue - returns - cogs),
      };
    });
  }, [agg, bizRows, effRange.from, effRange.to, effGranularity]);

  // Tổng hợp chỉ số kinh doanh cho TOÀN KỲ ĐANG CHỌN (đầu vào các thẻ KPI)
  const periodBiz = useMemo(() => {
    let invoices = 0, revenue = 0, returns = 0, cogs = 0;
    for (const p of bizRows) {
      if (p.date < effRange.from || p.date > effRange.to) continue;
      invoices += p.invoices ?? 0;
      revenue += p.revenue;
      returns += p.returns;
      cogs += p.cogs ?? 0;
    }
    return { invoices, revenue, returns, net: revenue - returns, cogs, gross: revenue - returns - cogs };
  }, [bizRows, effRange.from, effRange.to]);

  const top10Bars = useMemo(() =>
    agg.topFamilies.slice(0, 10).map(([name, value]) => ({ name, value: Math.round(value) })),
  [agg]);

  // Thị phần Top 4 nhóm hàng — tính theo kỳ đang chọn (thay API family-mix all-time)
  const mixPie = useMemo(() => {
    const top = agg.topFamilies.slice(0, 4);
    const total = agg.topFamilies.reduce((s, [, v]) => s + v, 0);
    return top.map(([name, value]) => ({
      name,
      value: Math.round(value),
      share: total > 0 ? (value / total) * 100 : 0,
    }));
  }, [agg]);

  // Chi tiết một ngày cụ thể
  const dayDetail = useMemo(() => {
    if (!specificDate) return null;
    const idx = allDates.indexOf(specificDate);
    if (idx === -1) return { found: false as const, total: 0, prevDiff: null as number | null, avg7Diff: null as number | null, bars: [] as { name: string; value: number }[] };
    const total = dailyTotals[specificDate];
    const prevTotal = idx > 0 ? dailyTotals[allDates[idx - 1]] : undefined;
    const prevDiff = prevTotal && prevTotal > 0 ? ((total - prevTotal) / prevTotal) * 100 : null;
    const prev7 = allDates.slice(Math.max(0, idx - 7), idx).map((d) => dailyTotals[d]);
    const avg7 = prev7.length ? prev7.reduce((s, v) => s + v, 0) / prev7.length : 0;
    const avg7Diff = avg7 > 0 ? ((total - avg7) / avg7) * 100 : null;
    const famMap: Record<string, number> = {};
    for (const r of histRows) {
      if (r.date !== specificDate) continue;
      famMap[r.family] = (famMap[r.family] || 0) + (Number(r.unit_sales) || 0);
    }
    const bars = Object.entries(famMap)
      .map(([name, value]) => ({ name, value: Math.round(value) }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);
    return { found: true as const, total, prevDiff, avg7Diff, bars };
  }, [specificDate, allDates, dailyTotals, histRows]);

  // Gợi ý tự động từ dữ liệu lịch sử trong khoảng đang chọn
  const insights = useMemo(() => {
    if (!agg.series.length) return [];
    const out: { tone: "good" | "warn" | "bad" | "info"; text: string }[] = [];

    // 1. Nửa sau vs nửa đầu của kỳ theo từng nhóm hàng
    const half = Math.floor(agg.series.length / 2);
    if (half >= 1 && agg.series.length >= 2) {
      const firstHalf: Record<string, number> = {};
      const secondHalf: Record<string, number> = {};
      agg.series.forEach((s, i) => {
        const target = i < half ? firstHalf : secondHalf;
        for (const [fam, v] of Object.entries(s.fams)) target[fam] = (target[fam] || 0) + v;
      });
      let best: { fam: string; pct: number } | null = null;
      let worst: { fam: string; pct: number } | null = null;
      for (const [fam, v2] of Object.entries(secondHalf)) {
        const v1 = firstHalf[fam] || 0;
        if (v1 < 500) continue;
        const pct = ((v2 - v1) / v1) * 100;
        if (!best || pct > best.pct) best = { fam, pct };
        if (!worst || pct < worst.pct) worst = { fam, pct };
      }
      if (best && best.pct > 5) out.push({ tone: "good", text: `"${best.fam}" đang tăng tốc (+${best.pct.toFixed(1)}% nửa sau so với nửa đầu của kỳ) — nên chủ động bổ sung tồn kho cho nhóm này.` });
      if (worst && worst.pct < -5) out.push({ tone: "bad", text: `"${worst.fam}" suy giảm ${worst.pct.toFixed(1)}% ở nửa sau của kỳ — cân nhắc giảm tạm nhịp nhập hàng để tránh tồn đọng.` });
    }

    // 2. Ngày đỉnh trong khoảng
    let peakD = ""; let peakV = -1;
    for (const d of allDates) {
      if (d < effRange.from || d > effRange.to) continue;
      if (dailyTotals[d] > peakV) { peakV = dailyTotals[d]; peakD = d; }
    }
    if (peakD) out.push({ tone: "info", text: `Ngày doanh số cao nhất kỳ: ${formatDateVN(peakD)} (${fmt(peakV)} đơn vị) — kiểm tra sẵn nhân sự và kho cho các ngày cao điểm tương tự.` });

    // 3. Tỷ trọng hàng dễ hỏng
    const totP = agg.series.reduce((s, x) => s + x.perishable, 0);
    const totAll = agg.series.reduce((s, x) => s + x.total, 0);
    if (totAll > 0) {
      const shareP = (totP / totAll) * 100;
      if (shareP >= 30) out.push({ tone: "warn", text: `Hàng dễ hỏng chiếm ${shareP.toFixed(1)}% doanh số kỳ này — cần luân chuyển nhanh, ưu tiên dự báo ngắn hạn cho PRODUCE/DAIRY.` });
    }

    // 4. Mức độ phụ thuộc nhóm dẫn đầu
    if (agg.topFamilies.length) {
      const [topFam, topVal] = agg.topFamilies[0];
      const totalAllFam = agg.topFamilies.reduce((s, [, v]) => s + v, 0);
      const share = (topVal / totalAllFam) * 100;
      if (share >= 40) out.push({ tone: "warn", text: `"${topFam}" chiếm ${share.toFixed(1)}% tổng doanh số — rủi ro phụ thuộc một nhóm hàng, nên đa dạng danh mục.` });
      else out.push({ tone: "info", text: `Nhóm dẫn đầu "${topFam}" chiếm ${share.toFixed(1)}% tổng doanh số kỳ này.` });
    }

    // 5. Xu hướng cuối kỳ
    const last = agg.series[agg.series.length - 1];
    const prevLast = agg.series[agg.series.length - 2];
    if (last && prevLast && last.growth !== null) {
      out.push({
        tone: last.growth >= 0 ? "good" : "bad",
        text: `Kỳ ${last.label} ${last.growth >= 0 ? "tăng" : "giảm"} ${Math.abs(last.growth).toFixed(1)}% so với kỳ ${prevLast.label} — ${last.growth >= 0 ? "duy trì nhịp nhập hàng hiện tại." : "xem xét khuyến mãi ngắn hạn để kéo lại nhu cầu."}`,
      });
    }

    return out.slice(0, 6);
  }, [agg, allDates, dailyTotals, effRange.from, effRange.to]);

  if (loading) {
    return (
      <div className="space-y-5">
        <SectionHeader title="Phân tích sản phẩm" sub="Hiệu suất và xu hướng danh mục sản phẩm" />
        <div className="flex items-center justify-center h-64 text-slate-500 text-sm gap-2 bg-white border border-slate-200 rounded-xl">
          <RefreshCw size={15} className="animate-spin" />
          Đang tải dữ liệu từ backend...
        </div>
      </div>
    );
  }

  if (err) {
    return (
      <div className="space-y-5">
        <SectionHeader title="Phân tích sản phẩm" sub="Hiệu suất và xu hướng danh mục sản phẩm" />
        <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-3 text-sm text-red-600">{err}</div>
      </div>
    );
  }

  const histTotal = agg.series.reduce((s, x) => s + x.total, 0);
  const granLabel = effGranularity === "day" ? "ngày" : effGranularity === "month" ? "tháng" : "quý";

  return (
    <div className="space-y-5">
      <SectionHeader title="Phân tích sản phẩm" sub="Hiệu suất danh mục — Dự báo LightGBM (backend) + Doanh số thực tế 2016–2017 (dataset)" />

      {/* Bộ điều khiển thời gian — áp dụng cho toàn bộ các bảng/dữ liệu thực tế bên dưới */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center gap-x-4 gap-y-3">
        <div className="flex gap-1.5">
          {([["day", "Ngày"], ["month", "Tháng"], ["quarter", "Quý"]] as [HistGranularity, string][]).map(([k, l]) => (
            <button
              key={k}
              onClick={() => setGranularity(k)}
              className={`text-xs px-3 py-2 rounded-lg border transition-colors ${granularity === k ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-white border-slate-200 text-slate-500 hover:text-slate-900"}`}
            >
              {l}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-xs text-slate-500">
          Từ ngày
          <input type="date" value={fromDate} min={bounds.min || undefined} max={bounds.max || undefined}
            onChange={(e) => setFromDate(e.target.value)}
            disabled={!bounds.min}
            className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-500/40 disabled:opacity-40" />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-500">
          Đến ngày
          <input type="date" value={toDate} min={bounds.min || undefined} max={bounds.max || undefined}
            onChange={(e) => setToDate(e.target.value)}
            disabled={!bounds.min}
            className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-500/40 disabled:opacity-40" />
        </label>
        {(fromDate || toDate) && (
          <button
            onClick={() => { setFromDate(""); setToDate(""); }}
            className="text-xs text-slate-400 hover:text-blue-600 underline underline-offset-2 transition-colors"
          >
            Cả kỳ
          </button>
        )}
        <div className="h-5 w-px bg-slate-200 hidden sm:block" />
        <label className="flex items-center gap-1.5 text-xs text-slate-500">
          <Zap size={12} className="text-amber-500 shrink-0" />
          Ngày cụ thể
          <input type="date" value={specificDate} min={bounds.min || undefined} max={bounds.max || undefined}
            onChange={(e) => setSpecificDate(e.target.value)}
            disabled={!bounds.min}
            className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-500/40 disabled:opacity-40" />
        </label>
        {specificDate && (
          <button onClick={() => setSpecificDate("")}
            className="text-xs text-slate-400 hover:text-red-500 underline underline-offset-2 transition-colors">
            Xóa
          </button>
        )}
      </div>

      {/* KPI chỉ số kinh doanh theo kỳ đang chọn (từ dữ liệu backend) — 2 hàng x 3 thẻ */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard
          label="Số hóa đơn"
          value={bizRows.length ? fmt(periodBiz.invoices) : "—"}
          sub="Giao dịch ghi nhận trong kỳ" trend="up" color="blue"
        />
        <StatCard
          label="Doanh thu"
          value={bizRows.length ? fmtMoney(periodBiz.revenue) : "—"}
          sub={`Kỳ ${effRange.from ? `${formatDateVN(effRange.from)}${effRange.to !== effRange.from ? ` → ${formatDateVN(effRange.to)}` : ""}` : "—"}`} trend="up" color="green"
        />
        <StatCard
          label="Giá trị trả"
          value={bizRows.length ? fmtMoney(periodBiz.returns) : "—"}
          sub={`${periodBiz.revenue > 0 ? ((periodBiz.returns / periodBiz.revenue) * 100).toFixed(1) : "0"}% doanh thu`} trend="down" color="red"
        />
        <StatCard
          label="Doanh thu thuần"
          value={bizRows.length ? fmtMoney(periodBiz.net) : "—"}
          sub="Doanh thu − Giá trị trả" trend="up" color="purple"
        />
        <StatCard
          label="Tổng giá vốn"
          value={bizRows.length ? fmtMoney(periodBiz.cogs) : "—"}
          sub="Giá vốn hàng bán ròng" trend="up" color="amber"
        />
        <StatCard
          label="Lợi nhuận gộp"
          value={bizRows.length ? fmtMoney(periodBiz.gross) : "—"}
          sub={`Biên ${periodBiz.net > 0 ? ((periodBiz.gross / periodBiz.net) * 100).toFixed(1) : "0"}% trên DT thuần`} trend="up" color="blue"
        />
      </div>

      {/* Xu hướng dự báo theo danh mục (backend) + chọn cửa sổ dự báo */}
      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <p className="text-sm font-semibold text-slate-900">Xu hướng dự báo theo danh mục</p>
            <p className="text-[10px] font-mono text-slate-500 mt-0.5">
              Đơn vị: số lượng bán / ngày — model LightGBM{horizon ? `, kỳ ${horizon.from} → ${(winLastDate || horizon.to)}` : ""}
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono text-slate-400 uppercase mr-1">Cửa sổ</span>
            {[7, 14, 21, 31].map((d) => (
              <button
                key={d}
                onClick={() => setTrendDays(d)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${trendDays === d ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-white border-slate-200 text-slate-500 hover:text-slate-900"}`}
              >
                {d} ngày
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={trendRows} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <defs>
              {families.map((f, i) => (
                <linearGradient key={f} id={`grad-${i}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={CHART_COLORS[i % CHART_COLORS.length]} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
            <XAxis dataKey="date" tickFormatter={(v: string) => (v || "").slice(5)}
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false}
              tickFormatter={(v) => v >= 1000 ? `${v / 1000}B` : `${v}`} />
            <Tooltip content={<ChartTip />} />
            <Legend wrapperStyle={{ fontSize: "10px", color: "#64748b", fontFamily: "JetBrains Mono", paddingTop: 8 }} />
            {families.map((f, i) => (
              <Area key={f} type="monotone" dataKey={f} name={f}
                stroke={CHART_COLORS[i % CHART_COLORS.length]} fill={`url(#grad-${i})`} strokeWidth={2} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Hàng biểu đồ lịch sử 1: chỉ số kinh doanh + top 10 nhóm hàng */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div className="xl:col-span-3 bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Chỉ số kinh doanh theo thời gian</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-4">
            Thang {granLabel}{!specificDate && customSpanDays <= 61 ? " (tự động)" : ""} · {effRange.from ? `${formatDateVN(effRange.from)} → ${formatDateVN(effRange.to)}` : "—"} · Quy đổi VND (tỷ giá tham chiếu 1 USD = 25.500₫) — ước tính từ số lượng bán × giá tham chiếu
          </p>
          {!bounds.min || bizState === "loading" ? (
            <div className="h-[260px] flex items-center justify-center text-slate-400 text-xs">Đang tải dữ liệu chỉ số kinh doanh...</div>
          ) : trendChartData.length === 0 ? (
            <div className="h-[260px] flex items-center justify-center text-slate-400 text-xs">Không có dữ liệu trong khoảng đã chọn.</div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={trendChartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="label" interval="preserveStartEnd"
                  tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <YAxis tickFormatter={fmtMoneyCompact}
                  tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <Tooltip content={<BizTip />} />
                <Legend wrapperStyle={{ fontSize: "10px", color: "#64748b", fontFamily: "JetBrains Mono", paddingTop: 8 }} />
                <Line type="monotone" dataKey="revenue" name="Doanh thu" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="returns" name="Trả hàng" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="grossProfit" name="Lợi nhuận gộp" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="xl:col-span-2 bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Top 10 nhóm hàng bán chạy</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-3">Trong khoảng thời gian đang chọn</p>
          {top10Bars.length === 0 ? (
            <div className="h-[300px] flex items-center justify-center text-slate-400 text-xs">Không có dữ liệu.</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={top10Bars} layout="vertical" margin={{ top: 0, right: 24, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9, fontFamily: "JetBrains Mono" }}
                  axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} />
                <YAxis type="category" dataKey="name" width={140}
                  tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTip />} cursor={{ fill: "rgba(59,130,246,0.06)" }} />
                <Bar dataKey="value" name="Doanh số" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={14} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Chi tiết ngày cụ thể */}
      {specificDate && (
        <div className="space-y-4">
          <p className="text-sm font-semibold text-slate-900 flex items-center gap-2">
            <Zap size={13} className="text-amber-500" />
            Chi tiết ngày {formatDateVN(specificDate)}
          </p>
          {!dayDetail || !dayDetail.found ? (
            <div className="bg-amber-500/10 border border-amber-500/25 rounded-lg px-4 py-3 text-sm text-amber-600">
              Không có dữ liệu bán hàng cho ngày này{bounds.min ? ` (dataset: ${formatDateVN(bounds.min)} → ${formatDateVN(bounds.max)})` : ""}.
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <StatCard label={`Tổng doanh số ${formatDateVN(specificDate)}`} value={fmt(dayDetail.total)}
                  sub="Toàn bộ ngành hàng" trend="up" color="purple" />
                <StatCard label="So với ngày trước đó"
                  value={dayDetail.prevDiff != null ? `${dayDetail.prevDiff >= 0 ? "+" : ""}${dayDetail.prevDiff.toFixed(1)}%` : "—"}
                  sub="Ngày có dữ liệu liền trước" trend={dayDetail.prevDiff == null || dayDetail.prevDiff >= 0 ? "up" : "down"} color="blue" />
                <StatCard label="So với TB 7 ngày trước"
                  value={dayDetail.avg7Diff != null ? `${dayDetail.avg7Diff >= 0 ? "+" : ""}${dayDetail.avg7Diff.toFixed(1)}%` : "—"}
                  sub="Trung bình 7 ngày có dữ liệu trước đó" trend={dayDetail.avg7Diff == null || dayDetail.avg7Diff >= 0 ? "up" : "down"} color="green" />
              </div>
              <div className="bg-white border border-slate-200 rounded-xl p-5">
                <p className="text-sm font-semibold text-slate-900 mb-1">Top ngành hàng trong ngày</p>
                <p className="text-[10px] font-mono text-slate-500 mb-3">Đơn vị: số lượng bán</p>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={dayDetail.bars} layout="vertical" margin={{ top: 0, right: 24, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                    <XAxis type="number" tick={{ fill: "#64748b", fontSize: 9, fontFamily: "JetBrains Mono" }}
                      axisLine={false} tickLine={false} tickFormatter={(v) => fmt(v)} />
                    <YAxis type="category" dataKey="name" width={140}
                      tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<ChartTip />} cursor={{ fill: "rgba(168,85,247,0.06)" }} />
                    <Bar dataKey="value" name="Doanh số" fill="#a855f7" radius={[0, 4, 4, 0]} barSize={14} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      )}

      {/* Gợi ý tự động */}
      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <p className="text-sm font-semibold text-slate-900 mb-1 flex items-center gap-2">
          <Zap size={13} className="text-amber-500" />
          Gợi ý từ dữ liệu
        </p>
        <p className="text-[10px] font-mono text-slate-500 mb-4">Tự động phân tích theo khoảng thời gian đang chọn</p>
        <div className="space-y-2.5">
          {insights.length === 0 ? (
            <p className="text-xs text-slate-500">Chưa đủ dữ liệu để đưa ra gợi ý.</p>
          ) : insights.map((it, i) => (
            <InsightCard key={i} tone={it.tone}>{it.text}</InsightCard>
          ))}
        </div>
      </div>

      {/* Thị phần danh mục — theo kỳ đang chọn */}
      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <p className="text-sm font-semibold text-slate-900">Thị phần danh mục (Top 4 nhóm hàng)</p>
        <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-3">
          Theo kỳ đang chọn: {effRange.from ? `${formatDateVN(effRange.from)} → ${formatDateVN(effRange.to)}` : "—"}{branchId !== "all" ? ` — Cửa hàng #${branchId}` : ""}
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={mixPie} cx="50%" cy="50%" innerRadius={38} outerRadius={70} paddingAngle={3} dataKey="value" nameKey="name">
                {mixPie.map((_, i) => (<Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />))}
              </Pie>
              <Tooltip content={<ChartTip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5">
            {mixPie.length === 0 ? (
              <p className="text-slate-500 text-xs">Chưa có dữ liệu thị phần trong kỳ này.</p>
            ) : mixPie.map((d, i) => (
              <div key={d.name} className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                  <span className="text-[11px] text-slate-700">{d.name}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">{d.share.toFixed(1)}% · {d.value.toLocaleString("vi-VN")} đơn vị</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Section: Quản lý sản phẩm (dữ liệu thật từ retail.db) ───── */
function ProductManagement({ branchId, apiBase, auth, onAuthError }: {
  branchId: string; apiBase: string; auth: AuthInfo; onAuthError: () => void;
}) {
  const [products, setProducts] = useState<ProductItem[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [lowStockCount, setLowStockCount] = useState(0);
  // Ngưỡng cảnh báo tồn kho do backend tính (30 × số cửa hàng trong phạm vi);
  // fallback heuristic cũ nếu backend chưa trả trường này.
  const [lowStockThreshold, setLowStockThreshold] = useState<number | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 15;
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [family, setFamily] = useState("all");
  const [families, setFamilies] = useState<string[]>([]);
  const [filter, setFilter] = useState<"all" | "active" | "outofstock">("all");
  const [sortKey, setSortKey] = useState<"sold" | "stock" | "name" | "item">("sold");
  const [sortAsc, setSortAsc] = useState(false);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  // Nhập hàng (restock) — chỉ khi đang chọn 1 cửa hàng cụ thể
  const [restockTarget, setRestockTarget] = useState<ProductItem | null>(null);
  const [restockQty, setRestockQty] = useState("50");
  const [restockMsg, setRestockMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [restocking, setRestocking] = useState(false);

  const storeParam = branchId !== "all" ? `&store_nbr=${branchId}` : "";
  const isRealStore = branchId === "all" || /^\d+$/.test(branchId);
  // Ngưỡng tô màu tồn kho thấp - ưu tiên giá trị backend trả về
  const stockWarnThreshold = lowStockThreshold ?? (branchId !== "all" ? 30 : 600);

  // Danh sách nhóm hàng cho dropdown lọc
  useEffect(() => {
    apiFetch(apiBase, "/api/product-families", auth)
      .then((r) => (r.ok ? r.json() : []))
      .then((d) => Array.isArray(d) && setFamilies(d))
      .catch(() => {});
  }, [apiBase, auth]);

  // Load danh sách sản phẩm (server-side search / filter / sort / paginate)
  useEffect(() => {
    if (!isRealStore) return;
    let cancelled = false;
    setLoading(true);
    setErr("");
    const params = `?page=${page}&page_size=${pageSize}` +
      (search ? `&search=${encodeURIComponent(search)}` : "") +
      (family !== "all" ? `&family=${encodeURIComponent(family)}` : "") +
      (filter !== "all" ? `&status=${filter}` : "") +
      `&sort=${sortKey}&order=${sortAsc ? "asc" : "desc"}${storeParam}`;
    apiFetch(apiBase, `/api/products${params}`, auth)
      .then((res) => {
        if (res.status === 401) { onAuthError(); return Promise.reject(); }
        return res.ok ? res.json() : res.json().then((d) => { throw new Error(d?.detail || `Lỗi ${res.status}`); });
      })
      .then((data: ProductsResponse) => {
        if (cancelled) return;
        setProducts(data.items || []);
        setTotal(data.total);
        setTotalPages(data.total_pages);
        setLowStockCount(data.low_stock_count ?? 0);
        setLowStockThreshold(data.low_stock_threshold ?? null);
      })
      .catch((e) => { if (!cancelled) setErr(e?.message || "Không tải được danh sách sản phẩm."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [page, search, family, filter, sortKey, sortAsc, apiBase, auth, isRealStore, onAuthError, branchId]);

  // Debounce ô tìm kiếm -> reset về trang 1
  useEffect(() => {
    const t = setTimeout(() => {
      if (searchInput.trim() !== search) {
        setSearch(searchInput.trim());
        setPage(1);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [searchInput, search]);

  const toggleSort = (k: typeof sortKey) => {
    if (sortKey === k) setSortAsc((x) => !x);
    else { setSortKey(k); setSortAsc(k === "name"); }
    setPage(1);
  };

  const submitRestock = async () => {
    if (!restockTarget || restocking) return;
    const qty = parseInt(restockQty, 10);
    if (!Number.isFinite(qty) || qty <= 0) {
      setRestockMsg({ ok: false, text: "Số lượng không hợp lệ." });
      return;
    }
    const targetStore = branchId !== "all" ? parseInt(branchId, 10) : NaN;
    if (!Number.isFinite(targetStore)) return;
    setRestocking(true);
    setRestockMsg(null);
    try {
      const res = await apiFetch(apiBase, "/api/inventory/restock", auth, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ store_nbr: targetStore, item_nbr: restockTarget.item_nbr, quantity: qty }),
      });
      if (res.status === 401) { onAuthError(); return; }
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `Lỗi ${res.status}`);
      setRestockMsg({ ok: true, text: data?.message || `Đã nhập ${qty} đơn vị cho #${restockTarget.item_nbr}.` });
      // Làm mới lại trang hiện tại để thấy tồn kho mới
      setLoading(true);
      apiFetch(apiBase, `/api/products?page=${page}&page_size=${pageSize}` +
        (search ? `&search=${encodeURIComponent(search)}` : "") +
        (family !== "all" ? `&family=${encodeURIComponent(family)}` : "") +
        (filter !== "all" ? `&status=${filter}` : "") +
        `&sort=${sortKey}&order=${sortAsc ? "asc" : "desc"}${storeParam}`, auth)
        .then((r) => r.json())
        .then((d: ProductsResponse) => {
          setProducts(d.items || []);
          setTotal(d.total);
          setTotalPages(d.total_pages);
          setLowStockCount(d.low_stock_count ?? 0);
          setLowStockThreshold(d.low_stock_threshold ?? null);
        })
        .finally(() => setLoading(false));
    } catch (e: any) {
      setRestockMsg({ ok: false, text: e?.message || "Không nhập hàng được." });
    } finally {
      setRestocking(false);
    }
  };

  const statusBadge = (s: string) => {
    const map: Record<string, { label: string; cls: string }> = {
      active: { label: "Đang bán", cls: "bg-emerald-500/15 text-emerald-600 border-emerald-500/25" },
      outofstock: { label: "Hết hàng", cls: "bg-red-500/15 text-red-600 border-red-500/25" },
    };
    const cfg = map[s] || map.active;
    return <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border whitespace-nowrap ${cfg.cls}`}>{cfg.label}</span>;
  };

  const SortIcon = ({ k }: { k: typeof sortKey }) =>
    sortKey === k ? (sortAsc ? <ChevronUp size={12} className="text-blue-600" /> : <ChevronDown size={12} className="text-blue-600" />) : <ChevronUp size={12} className="text-slate-300" />;

  return (
    <div className="space-y-5">
      <SectionHeader
        title="Quản lý sản phẩm"
        sub={`${total.toLocaleString("vi-VN")} sản phẩm trong danh mục${branchId !== "all" ? ` — Cửa hàng #${branchId}` : " — Toàn hệ thống"} · Nguồn: retail.db`}
      />

      <div className="flex flex-wrap gap-2.5 items-center">
        <div className="relative max-w-xs w-full min-w-[200px] flex-1 sm:flex-none">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Tìm theo tên sản phẩm, mã SP, nhóm hàng..."
            className="w-full bg-white border border-slate-200 rounded-lg pl-8 pr-4 py-2 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500/40 transition-colors"
          />
        </div>
        <select
          value={family}
          onChange={(e) => { setFamily(e.target.value); setPage(1); }}
          className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-600 focus:outline-none focus:border-blue-500/40 max-w-[180px]"
        >
          <option value="all">Tất cả nhóm hàng</option>
          {families.map((f) => (<option key={f} value={f}>{f}</option>))}
        </select>
        <div className="flex gap-1.5 flex-wrap">
          {[{ key: "all", label: "Tất cả" }, { key: "active", label: "Đang bán" }, { key: "outofstock", label: "Hết hàng" }].map((f) => (
            <button
              key={f.key}
              onClick={() => { setFilter(f.key as typeof filter); setPage(1); }}
              className={`text-xs px-3 py-2 rounded-lg border transition-colors ${filter === f.key ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-white border-slate-200 text-slate-500 hover:text-slate-900"}`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {err && (
        <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-3 text-sm text-red-600">{err}</div>
      )}

      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/50">
                <th className="text-left px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                  <button className="flex items-center gap-1 hover:text-slate-900 transition-colors uppercase" onClick={() => toggleSort("item")}>Mã SP <SortIcon k="item" /></button>
                </th>
                <th className="text-left px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                  <button className="flex items-center gap-1 hover:text-slate-900 transition-colors uppercase" onClick={() => toggleSort("name")}>Tên sản phẩm <SortIcon k="name" /></button>
                </th>
                <th className="text-left px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Nhóm hàng</th>
                <th className="text-left px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Class</th>
                <th className="text-center px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Dễ hỏng</th>
                <th className="text-right px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                  <button className="flex items-center gap-1 ml-auto hover:text-slate-900 transition-colors uppercase" onClick={() => toggleSort("stock")}>Tồn kho <SortIcon k="stock" /></button>
                </th>
                <th className="text-right px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                  <button className="flex items-center gap-1 ml-auto hover:text-slate-900 transition-colors uppercase" onClick={() => toggleSort("sold")}>Bán 2016 <SortIcon k="sold" /></button>
                </th>
                <th className="text-right px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider" title="Tổng dự báo 16 ngày tới (toàn chuỗi) — dải tin cậy 1σ theo RMSLE của family">
                  Dự báo 16N
                </th>
                <th className="text-center px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider" title="A: nhóm chiếm 80% doanh số · B: đến 95% · C: dài đuôi">
                  Lớp ABC
                </th>
                <th className="text-center px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Trạng thái</th>
                {branchId !== "all" && <th className="px-4 py-3" />}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={branchId !== "all" ? 11 : 10} className="text-center py-8 text-slate-500 text-sm">
                  <RefreshCw size={14} className="animate-spin inline mr-2" />Đang tải...
                </td></tr>
              ) : products.length === 0 ? (
                <tr><td colSpan={branchId !== "all" ? 11 : 10} className="text-center py-8 text-slate-500 text-sm">Không có sản phẩm nào khớp bộ lọc.</td></tr>
              ) : products.map((p) => (
                <tr key={p.item_nbr} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3"><span className="text-xs font-mono font-medium text-slate-900">#{p.item_nbr}</span></td>
                  <td className="px-4 py-3">
                    <div className="max-w-[240px]">
                      <div className="text-xs text-slate-900 truncate" title={p.name ?? undefined}>{p.name ?? "—"}</div>
                      <div className="text-[10px] font-mono text-slate-400 sm:hidden">#{p.item_nbr}</div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-[10px] font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded-md whitespace-nowrap">{p.family}</span>
                  </td>
                  <td className="px-4 py-3"><span className="text-[10px] font-mono text-slate-500">{p.class_code ?? "—"}</span></td>
                  <td className="px-4 py-3 text-center">
                    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-md ${p.perishable === 1 ? "bg-red-500/10 text-red-600" : "bg-emerald-500/10 text-emerald-600"}`}>
                      {p.perishable === 1 ? "Có" : "Không"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`text-xs font-mono ${p.stock === 0 ? "text-red-500" : p.stock < stockWarnThreshold ? "text-orange-500" : "text-emerald-500"}`}>
                      {Math.round(p.stock).toLocaleString("vi-VN")}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-xs font-mono text-slate-700">{Math.round(p.sold_2016).toLocaleString("vi-VN")}</span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-xs font-mono text-purple-600">{p.fc_total_16d != null ? Math.round(p.fc_total_16d).toLocaleString("vi-VN") : "—"}</span>
                    {p.fc_low != null && p.fc_high != null && (
                      <p className="text-[9px] font-mono text-slate-400 mt-0.5">
                        {Math.round(p.fc_low).toLocaleString("vi-VN")}–{Math.round(p.fc_high).toLocaleString("vi-VN")}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {p.abc_class
                      ? <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${abcBadgeClass(p.abc_class)}`}>{p.abc_class}</span>
                      : <span className="text-[10px] font-mono text-slate-300">—</span>}
                  </td>
                  <td className="px-4 py-3 text-center">{statusBadge(p.status)}</td>
                  {branchId !== "all" && (
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1 justify-end">
                        <button
                          className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                          title={`Nhập thêm hàng cho cửa hàng #${branchId}`}
                          onClick={() => { setRestockTarget(p); setRestockQty("50"); setRestockMsg(null); }}
                        >
                          <PackagePlus size={13} />
                          Nhập hàng
                        </button>
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Phân trang */}
        <div className="px-5 py-3 border-t border-slate-200 flex items-center justify-between flex-wrap gap-2">
          <p className="text-slate-500 text-xs">
            Trang {page}/{totalPages} · {total.toLocaleString("vi-VN")} sản phẩm
            {lowStockCount > 0 && <> · <span className="text-orange-500">{lowStockCount.toLocaleString("vi-VN")} cần bổ sung tồn kho</span></>}
          </p>
          <div className="flex items-center gap-1.5">
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="p-1.5 rounded-md border border-slate-200 text-slate-500 hover:text-slate-900 disabled:opacity-30 transition-colors"
            ><ChevronLeft size={13} /></button>
            <span className="text-xs font-mono text-slate-500 w-14 text-center">{page} / {totalPages}</span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="p-1.5 rounded-md border border-slate-200 text-slate-500 hover:text-slate-900 disabled:opacity-30 transition-colors"
            ><ChevronRight size={13} /></button>
          </div>
        </div>
      </div>

      {/* Dialog nhập hàng */}
      {restockTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4" onClick={() => !restocking && setRestockTarget(null)}>
          <div className="bg-white rounded-xl border border-slate-200 shadow-2xl w-full max-w-sm p-5 space-y-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-slate-900" title={restockTarget.name ?? undefined}>
                  Nhập hàng — {restockTarget.name ?? `#${restockTarget.item_nbr}`}
                </p>
                <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                  #{restockTarget.item_nbr} · Cửa hàng #{branchId} · Tồn kho hiện tại: {Math.round(restockTarget.stock).toLocaleString("vi-VN")}
                </p>
              </div>
            </div>
            <label className="block">
              <span className="block text-[10px] font-mono text-slate-500 uppercase tracking-wider mb-1">Số lượng nhập thêm</span>
              <input
                type="number"
                min={1}
                value={restockQty}
                onChange={(e) => setRestockQty(e.target.value)}
                autoFocus
                onKeyDown={(e) => e.key === "Enter" && submitRestock()}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-900 focus:outline-none focus:border-blue-500/40"
              />
            </label>
            {restockMsg && (
              <div className={`rounded-lg px-3 py-2 text-[11px] ${restockMsg.ok ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/25" : "bg-red-500/10 text-red-600 border border-red-500/25"}`}>
                {restockMsg.text}
              </div>
            )}
            <div className="flex gap-2 justify-end">
              <button onClick={() => setRestockTarget(null)} disabled={restocking}
                className="px-3 py-2 text-xs text-slate-500 hover:text-slate-900 rounded-lg transition-colors">Đóng</button>
              <button onClick={submitRestock} disabled={restocking}
                className="flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white text-xs font-medium px-4 py-2 rounded-lg transition-colors">
                {restocking ? "Đang xử lý..." : "Xác nhận nhập"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Nav config ───────────────────────────────────────────────── */
type View = "sales" | "chatbot" | "analysis" | "products" | "scenario";

const navItems: { id: View; label: string; icon: ElementType }[] = [
  { id: "sales", label: "Dự báo doanh số", icon: TrendingUp },
  { id: "chatbot", label: "AI Chatbot", icon: Bot },
  { id: "analysis", label: "Phân tích sản phẩm", icon: BarChart2 },
  { id: "products", label: "Quản lý sản phẩm", icon: Package },
  { id: "scenario", label: "Kịch bản What-if", icon: FlaskConical },
];

/* ── Thanh chọn cơ sở: kéo ngang bằng chuột ──────────────────── */
type BranchOption = { id: string; label: string; city: string };

function useDragScroll() {
  const ref = useRef<HTMLDivElement>(null);
  const draggedRef = useRef(false); // true ngay sau một lần kéo -> chặn click chọn nhầm
  const [edges, setEdges] = useState({ left: false, right: false });

  const updateEdges = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    setEdges({
      left: el.scrollLeft > 4,
      right: el.scrollLeft + el.clientWidth < el.scrollWidth - 4,
    });
  }, []);

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el || e.pointerType === "touch") return; // touch dùng cuộn native
    const startX = e.clientX;
    const startScroll = el.scrollLeft;
    let moved = false;
    el.style.cursor = "grabbing";
    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      if (Math.abs(dx) > 5) moved = true;
      el.scrollLeft = startScroll - dx;
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      draggedRef.current = moved;
      el.style.cursor = "";
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, []);

  // Bánh xe chuột dọc -> cuộn ngang khi rê vào thanh
  const onWheel = useCallback((e: ReactWheelEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el || !e.deltaY || e.deltaX) return;
    el.scrollLeft += e.deltaY;
  }, []);

  const scrollByAmount = useCallback((dir: number) => {
    ref.current?.scrollBy({ left: dir * 260, behavior: "smooth" });
  }, []);

  return { ref, draggedRef, edges, updateEdges, onPointerDown, onWheel, scrollByAmount };
}

function BranchBar({ branches, branchId, onSelect }: {
  branches: BranchOption[];
  branchId: string;
  onSelect: (id: string) => void;
}) {
  const { ref, draggedRef, edges, updateEdges, onPointerDown, onWheel, scrollByAmount } = useDragScroll();

  useEffect(() => {
    updateEdges();
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(updateEdges);
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [branches.length]);

  return (
    <div className="border-b border-slate-200 bg-white flex items-stretch">
      <button
        onClick={() => scrollByAmount(-1)}
        disabled={!edges.left}
        title="Cuộn sang trái"
        className="px-1.5 text-slate-400 hover:text-slate-900 disabled:opacity-30 disabled:hover:text-slate-400 transition-colors shrink-0"
      >
        <ChevronLeft size={14} />
      </button>
      <div
        ref={ref}
        onPointerDown={onPointerDown}
        onWheel={onWheel}
        onScroll={updateEdges}
        className="flex-1 min-w-0 overflow-x-auto scrollbar-thin-x cursor-grab active:cursor-grabbing select-none"
      >
        <div className="flex items-center gap-1.5 min-w-max py-2 px-1">
          <Building2 size={13} className="text-slate-400 mr-1 shrink-0" />
          {branches.map((b) => {
            const isActive = branchId === b.id;
            return (
              <button
                key={b.id}
                onClick={() => { if (!draggedRef.current) onSelect(b.id); }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs whitespace-nowrap transition-all ${isActive ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-white border-slate-200 text-slate-500 hover:text-slate-900 hover:border-slate-300"}`}
              >
                {b.id !== "all" && <MapPin size={10} className="shrink-0" />}
                {b.label}
                {isActive && b.id !== "all" && (<span className="text-[9px] font-normal text-blue-400 ml-0.5 hidden sm:inline">{b.city}</span>)}
              </button>
            );
          })}
        </div>
      </div>
      <button
        onClick={() => scrollByAmount(1)}
        disabled={!edges.right}
        title="Cuộn sang phải"
        className="px-1.5 text-slate-400 hover:text-slate-900 disabled:opacity-30 disabled:hover:text-slate-400 transition-colors shrink-0"
      >
        <ChevronRight size={14} />
      </button>
    </div>
  );
}

/* ── Section: Kịch bản What-if (Scenario Lab) ─────────────────── */
interface ScenarioMeta {
  store_nbr: number;
  family: string;
  future_dates: string[];
  default_stock: number;
  default_lead_time: number;
  sku_count: number;
  unit_price: number;
  baseline_promo_days: number;
  current_oil_price: number | null;
  last_transactions: number;
}

interface ScenarioKpi {
  baseline_total: number;
  scenario_total: number;
  delta: number;
  delta_pct: number | null;
  stock: number;
  days_of_cover: number;
  shortfall: number;
  excess: number;
  will_stockout: boolean;
  overstock: boolean;
  risk_level: string;
  verdict: string;
  unit_price: number;
  expected_revenue_usd: number;
  baseline_revenue_usd: number;
  lost_revenue_usd: number;
}

interface ScenarioSkuRow {
  item_nbr: number;
  baseline_total: number;
  scenario_total: number;
  delta: number;
  delta_pct: number | null;
  stock: number;
  shortfall: number;
}

interface ScenarioResult {
  status: string;
  source: string;
  store_nbr: number;
  family: string;
  horizon_days: number;
  future_dates: string[];
  baseline_series: Array<{ date: string; predicted_sales: number }>;
  scenario_series: Array<{ date: string; predicted_sales: number }>;
  sku_count: number;
  inventory: { stock: number; lead_time: number; sku_count: number; stock_used: number; lead_time_used: number };
  kpi: ScenarioKpi;
  analysis: string;
  recommendation: string;
  top_sku_movements: ScenarioSkuRow[];
}

function ScenarioLab({ branchId, apiBase, auth, onAuthError }: {
  branchId: string; apiBase: string; auth: AuthInfo; onAuthError: () => void;
}) {
  const [stores, setStores] = useState<number[]>([]);
  const [store, setStore] = useState<number | null>(/^\d+$/.test(branchId) ? Number(branchId) : null);
  const [families, setFamilies] = useState<string[]>([]);
  const [family, setFamily] = useState("");
  const [meta, setMeta] = useState<ScenarioMeta | null>(null);
  const [metaErr, setMetaErr] = useState("");

  // Các núm chỉnh kịch bản ("" = dùng giá trị thật)
  const [multiplier, setMultiplier] = useState(1);
  const [promoMode, setPromoMode] = useState<"real" | "custom">("real");
  const [promoDays, setPromoDays] = useState(8);
  const [oil, setOil] = useState("");
  const [trafficOn, setTrafficOn] = useState(false);
  const [traffic, setTraffic] = useState(20);
  const [eventType, setEventType] = useState<"none" | "holiday" | "earthquake">("none");
  const [eventDays, setEventDays] = useState(3);
  const [stock, setStock] = useState("");
  const [leadTime, setLeadTime] = useState("");

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [err, setErr] = useState("");

  // Danh sách cửa hàng + ngành hàng (lấy 1 lần)
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      apiFetch(apiBase, "/api/stores", auth),
      apiFetch(apiBase, "/api/product-families", auth),
    ])
      .then(([sRes, fRes]) => {
        if (sRes.status === 401 || fRes.status === 401) { onAuthError(); return Promise.reject(); }
        return Promise.all([sRes.json(), fRes.ok ? fRes.json() : Promise.resolve([])]);
      })
      .then(([sList, fList]) => {
        if (cancelled) return;
        if (Array.isArray(sList)) setStores(sList);
        if (Array.isArray(fList)) setFamilies(fList);
      })
      .catch(() => { /* để trống dropdown, user vẫn thấy lỗi khi chạy */ });
    return () => { cancelled = true; };
  }, [apiBase, auth, onAuthError]);

  // Chọn cửa hàng mặc định theo tab cơ sở đang mở
  useEffect(() => {
    if (store === null && stores.length > 0) {
      const preferred = /^\d+$/.test(branchId) ? Number(branchId) : NaN;
      setStore(Number.isFinite(preferred) && stores.includes(preferred) ? preferred : stores[0]);
    }
  }, [stores, store, branchId]);

  // Meta mặc định cho form prefill theo (store, family)
  useEffect(() => {
    if (!store || !family) { setMeta(null); return; }
    let cancelled = false;
    setMetaErr("");
    apiFetch(apiBase, `/api/scenario/meta?store_nbr=${store}&family=${encodeURIComponent(family)}`, auth)
      .then((res) => {
        if (res.status === 401) { onAuthError(); return Promise.reject(); }
        return res.ok ? res.json() : res.json().then((d) => { throw new Error(d?.detail || `Lỗi ${res.status}`); });
      })
      .then((m) => { if (!cancelled) setMeta(m); })
      .catch((e) => { if (!cancelled) { setMeta(null); setMetaErr(e?.message || "Không tải được thông tin mặc định."); } });
    return () => { cancelled = true; };
  }, [store, family, apiBase, auth, onAuthError]);

  async function runScenario() {
    if (!store || !family) return;
    setRunning(true);
    setErr("");
    setResult(null);
    try {
      const body: Record<string, unknown> = {
        store_nbr: store,
        family,
        demand_multiplier: Number(multiplier) || 1,
        event_type: eventType,
        event_days: eventType === "none" ? 0 : eventDays,
      };
      if (promoMode === "custom") body.promo_days = Number(promoDays) || 0;
      if (oil !== "" && oil !== null) body.oil_price = Number(oil);
      if (trafficOn) body.traffic_change_pct = Number(traffic) || 0;
      if (stock !== "") body.stock_override = Number(stock);
      if (leadTime !== "") body.lead_time_override = Number(leadTime);
      const res = await apiFetch(apiBase, "/api/scenario/run", auth, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 401) { onAuthError(); return; }
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(typeof data?.detail === "string" ? data.detail : `Lỗi ${res.status}`);
      setResult(data as ScenarioResult);
    } catch (e: any) {
      setErr(e?.message || "Không chạy được kịch bản.");
    } finally {
      setRunning(false);
    }
  }

  // Gộp 2 series theo ngày cho biểu đồ
  const chartData = useMemo(() => {
    if (!result) return [];
    const base = new Map(result.baseline_series.map((p) => [p.date, p.predicted_sales]));
    return result.scenario_series.map((p) => ({
      date: p.date.slice(5),
      hientai: Math.round(base.get(p.date) ?? 0),
      kichban: Math.round(p.predicted_sales),
    }));
  }, [result]);

  const kpi = result?.kpi;
  const isNeutral = Number(multiplier) === 1 && promoMode === "real" && oil === ""
    && !trafficOn && eventType === "none" && stock === "" && leadTime === "";

  const inputCls = "bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:border-blue-500/40 disabled:opacity-40";
  const knobLabel = "text-[10px] font-mono text-slate-500 uppercase tracking-wide";

  return (
    <div className="space-y-5">
      <SectionHeader title="Kịch bản What-if" sub="Sửa số liệu → dự báo lại bằng mô hình thật → phân tích tác động tức thì" />

      {/* Cấu hình kịch bản */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-4">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
          <label className="flex items-center gap-2 text-xs text-slate-500">
            <MapPin size={12} className="text-slate-400 shrink-0" /> Cửa hàng
            <select value={store ?? ""} onChange={(e) => setStore(Number(e.target.value))} className={inputCls}>
              {(stores.length ? stores : (store ? [store] : [])).map((s) => (
                <option key={s} value={s}>Cửa hàng #{s}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-500">
            <Package size={12} className="text-slate-400 shrink-0" /> Ngành hàng
            <select value={family} onChange={(e) => setFamily(e.target.value)} className={`${inputCls} max-w-[220px]`}>
              <option value="">— chọn ngành hàng —</option>
              {families.map((f) => (<option key={f} value={f}>{f}</option>))}
            </select>
          </label>
          {meta && (
            <span className="text-[10px] font-mono text-slate-400">
              {meta.sku_count} SKU · giá ref ${meta.unit_price}/đơn vị · KM baseline {meta.baseline_promo_days}/16 ngày
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-6 gap-y-4">
          {/* 1. Hệ số nhu cầu */}
          <div className="space-y-1.5">
            <p className={knobLabel}>1 · Hệ số nhu cầu thị trường</p>
            <div className="flex items-center gap-2">
              <input type="range" min={0.5} max={2} step={0.05} value={multiplier}
                onChange={(e) => setMultiplier(Number(e.target.value))} className="flex-1 accent-blue-600" />
              <span className="text-xs font-mono text-slate-700 w-14 text-right">×{multiplier.toFixed(2)}</span>
            </div>
            <p className="text-[10px] text-slate-400">1.00 = giữ nguyên · 1.50 = +50% · 0.80 = −20%</p>
          </div>

          {/* 2. Khuyến mãi */}
          <div className="space-y-1.5">
            <p className={knobLabel}>2 · Khuyến mãi (onpromotion)</p>
            <div className="flex gap-1.5">
              <button onClick={() => setPromoMode("real")}
                className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors ${promoMode === "real" ? "bg-blue-50 border-blue-200 text-blue-600" : "border-slate-200 text-slate-500"}`}>
                Lịch thật{meta ? ` (${meta.baseline_promo_days} ngày)` : ""}
              </button>
              <button onClick={() => setPromoMode("custom")}
                className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors ${promoMode === "custom" ? "bg-blue-50 border-blue-200 text-blue-600" : "border-slate-200 text-slate-500"}`}>
                Tùy chỉnh
              </button>
            </div>
            {promoMode === "custom" && (
              <div className="flex items-center gap-2">
                <input type="range" min={0} max={16} step={1} value={promoDays}
                  onChange={(e) => setPromoDays(Number(e.target.value))} className="flex-1 accent-blue-600" />
                <span className="text-xs font-mono text-slate-700 w-14 text-right">{promoDays} ngày</span>
              </div>
            )}
          </div>

          {/* 3. Giá dầu */}
          <div className="space-y-1.5">
            <p className={knobLabel}>3 · Giá dầu (USD)</p>
            <div className="flex items-center gap-2">
              <input type="number" min={20} max={200} step={0.5} value={oil}
                onChange={(e) => setOil(e.target.value)} placeholder={meta?.current_oil_price ? String(meta.current_oil_price) : "giá thật"}
                className={`${inputCls} w-28`} />
              {oil !== "" && (
                <button onClick={() => setOil("")} className="text-[10px] text-slate-400 hover:text-blue-600 underline underline-offset-2">Dùng giá thật</button>
              )}
            </div>
            <p className="text-[10px] text-slate-400">Để trống = giá dầu thật từ oil.csv</p>
          </div>

          {/* 4. Lưu lượng khách */}
          <div className="space-y-1.5">
            <p className={knobLabel}>4 · Lưu lượng khách</p>
            <div className="flex items-center gap-2">
              <button onClick={() => setTrafficOn(!trafficOn)}
                className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors ${trafficOn ? "bg-blue-50 border-blue-200 text-blue-600" : "border-slate-200 text-slate-500"}`}>
                {trafficOn ? "Đang chỉnh" : "Giữ nguyên"}
              </button>
              {trafficOn && (
                <>
                  <input type="range" min={-50} max={100} step={5} value={traffic}
                    onChange={(e) => setTraffic(Number(e.target.value))} className="flex-1 accent-blue-600" />
                  <span className="text-xs font-mono text-slate-700 w-14 text-right">{traffic > 0 ? "+" : ""}{traffic}%</span>
                </>
              )}
            </div>
          </div>

          {/* 5. Sự kiện bất ngờ */}
          <div className="space-y-1.5">
            <p className={knobLabel}>5 · Sự kiện bất ngờ</p>
            <div className="flex items-center gap-2">
              <select value={eventType} onChange={(e) => setEventType(e.target.value as typeof eventType)} className={inputCls}>
                <option value="none">Không có</option>
                <option value="holiday">Ngày lễ đột xuất</option>
                <option value="earthquake">Thiên tai (cú sốc nhu cầu)</option>
              </select>
              {eventType !== "none" && (
                <div className="flex items-center gap-1.5">
                  <input type="number" min={1} max={16} value={eventDays}
                    onChange={(e) => setEventDays(Math.max(1, Math.min(16, Number(e.target.value) || 1)))}
                    className={`${inputCls} w-16`} />
                  <span className="text-[10px] text-slate-400">ngày đầu kỳ</span>
                </div>
              )}
            </div>
          </div>

          {/* 6. Tồn kho + lead time */}
          <div className="space-y-1.5">
            <p className={knobLabel}>6 · Tồn kho & lead time (tầng phân tích)</p>
            <div className="flex items-center gap-2 flex-wrap">
              <input type="number" min={0} step={1} value={stock} onChange={(e) => setStock(e.target.value)}
                placeholder={meta ? `tồn thật ${Math.round(meta.default_stock)}` : "tồn thật"} className={`${inputCls} w-32`} />
              <input type="number" min={0} max={60} step={0.5} value={leadTime} onChange={(e) => setLeadTime(e.target.value)}
                placeholder={meta ? `LT ${meta.default_lead_time} ngày` : "lead time"} className={`${inputCls} w-24`} />
              {(stock !== "" || leadTime !== "") && (
                <button onClick={() => { setStock(""); setLeadTime(""); }}
                  className="text-[10px] text-slate-400 hover:text-blue-600 underline underline-offset-2">Dùng giá trị thật</button>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 pt-1">
          <button onClick={runScenario} disabled={running || !store || !family}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white text-xs font-medium px-4 py-2.5 rounded-lg transition-colors">
            {running ? <RefreshCw size={13} className="animate-spin" /> : <FlaskConical size={13} />}
            {running ? "Đang chạy dự báo..." : "Chạy kịch bản"}
          </button>
          {isNeutral && !running && (
            <span className="text-[10px] text-amber-500 bg-amber-500/10 border border-amber-500/20 rounded-lg px-2.5 py-1.5">
              Chưa chỉnh số liệu nào — kết quả sẽ xấp xỉ hiện tại, dùng để kiểm tra mô hình.
            </span>
          )}
        </div>
        {metaErr && (
          <p className="text-[11px] text-red-500 bg-red-500/10 border border-red-500/20 rounded-lg px-2.5 py-2">{metaErr}</p>
        )}
      </div>

      {/* Lỗi */}
      {err && (
        <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-3 text-sm text-red-600">{err}</div>
      )}

      {/* Kết quả */}
      {result && kpi && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard label="Dự báo kịch bản (tổng kỳ)" value={fmt(kpi.scenario_total)}
              sub={`Hiện tại: ${fmt(kpi.baseline_total)} · nguồn: ${result.source === "ml_service" ? "model thật" : "xấp xỉ"}`}
              trend={kpi.delta >= 0 ? "up" : "down"} color="purple" />
            <StatCard label="Thay đổi so với hiện tại"
              value={`${kpi.delta_pct !== null ? (kpi.delta_pct > 0 ? "+" : "") + kpi.delta_pct.toFixed(1) + "%" : "—"}`}
              sub={`${fmt(kpi.delta)} đơn vị`} trend={kpi.delta >= 0 ? "up" : "down"}
              color={kpi.delta >= 0 ? "green" : "amber"} />
            <StatCard label={kpi.will_stockout ? "Thiếu hàng (nếu không nhập)" : kpi.overstock ? "Dư tồn" : "Đủ hàng"}
              value={fmt(kpi.will_stockout ? kpi.shortfall : kpi.overstock ? kpi.excess : 0)}
              sub={`Tồn ${fmt(kpi.stock)} · phủ ~${kpi.days_of_cover} ngày`}
              trend={kpi.will_stockout ? "down" : "up"} color={kpi.will_stockout ? "amber" : "green"} />
            <StatCard label="Doanh thu kỳ vọng" value={`$${fmtMoney(kpi.expected_revenue_usd)}`}
              sub={kpi.will_stockout ? `Rủi ro mất ~$${fmtMoney(kpi.lost_revenue_usd)}` : `Hiện tại: $${fmtMoney(kpi.baseline_revenue_usd)}`}
              trend="up" color="blue" />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            <div className="xl:col-span-2 bg-white border border-slate-200 rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-sm font-semibold text-slate-900">Dự báo theo ngày: hiện tại vs kịch bản</p>
                  <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                    {result.family} · cửa hàng #{result.store_nbr} · {result.horizon_days} ngày
                  </p>
                </div>
                <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
                  <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-slate-400 rounded inline-block" />Hiện tại</span>
                  <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-blue-600 rounded inline-block" />Kịch bản</span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTip />} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line type="monotone" dataKey="hientai" name="Hiện tại" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 4" dot={false} />
                  <Line type="monotone" dataKey="kichban" name="Kịch bản" stroke="#2563eb" strokeWidth={2.5}
                    dot={{ fill: "#2563eb", r: 3, strokeWidth: 0 }} activeDot={{ r: 4 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="space-y-4">
              <div className={`border rounded-xl p-5 ${kpi.will_stockout ? "bg-red-500/5 border-red-500/25" : kpi.overstock ? "bg-amber-500/5 border-amber-500/25" : "bg-emerald-500/5 border-emerald-500/25"}`}>
                <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-2">
                  Kết luận · rủi ro {kpi.risk_level}
                </p>
                <p className="text-sm text-slate-800 whitespace-pre-line leading-relaxed">{result.analysis}</p>
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-xl p-5">
                <p className="text-[10px] font-mono uppercase tracking-widest text-blue-500 mb-2">Khuyến nghị hành động</p>
                <p className="text-sm text-slate-800 leading-relaxed">{result.recommendation}</p>
              </div>
            </div>
          </div>

          {/* Top SKU biến động */}
          {result.top_sku_movements.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl p-5">
              <div className="flex items-baseline justify-between mb-3">
                <p className="text-sm font-semibold text-slate-900">Top SKU biến động nhiều nhất</p>
                <p className="text-[9px] font-mono text-slate-400 uppercase">phân rã từ {result.sku_count} SKU theo ngành</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-[10px] font-mono text-slate-400 uppercase border-b border-slate-200">
                      <th className="py-2 pr-3">SKU</th>
                      <th className="py-2 pr-3 text-right">Hiện tại</th>
                      <th className="py-2 pr-3 text-right">Kịch bản</th>
                      <th className="py-2 pr-3 text-right">Δ</th>
                      <th className="py-2 pr-3 text-right">Tồn</th>
                      <th className="py-2 text-right">Thiếu</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.top_sku_movements.map((r) => (
                      <tr key={r.item_nbr} className="border-b border-slate-100 last:border-0">
                        <td className="py-2 pr-3 font-mono text-slate-700">#{r.item_nbr}</td>
                        <td className="py-2 pr-3 text-right font-mono text-slate-500">{fmt(r.baseline_total)}</td>
                        <td className="py-2 pr-3 text-right font-mono text-slate-800">{fmt(r.scenario_total)}</td>
                        <td className={`py-2 pr-3 text-right font-mono ${r.delta >= 0 ? "text-emerald-600" : "text-red-500"}`}>
                          {r.delta >= 0 ? "+" : ""}{fmt(r.delta)}
                        </td>
                        <td className="py-2 pr-3 text-right font-mono text-slate-500">{fmt(r.stock)}</td>
                        <td className={`py-2 text-right font-mono ${r.shortfall > 0 ? "text-red-500 font-medium" : "text-slate-400"}`}>
                          {fmt(r.shortfall)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-slate-400 mt-2">
                Phân rã SKU theo tỷ trọng dự báo hiện có — số liệu SKU mang tính xấp xỉ.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Shell components (kiểu frontend2) ───────────────────────── */
function Sidebar({
  view, setView, open, user, onLogout,
}: {
  view: View; setView: (v: View) => void; open: boolean;
  user: { displayName: string; role: string; username?: string } | null;
  onLogout: () => void;
}) {
  return (
    <aside
      className="flex-shrink-0 bg-white border-r border-slate-200 flex flex-col transition-all duration-200 overflow-hidden z-10"
      style={{ width: open ? "220px" : "52px" }}
    >
      <div className="h-12 flex items-center gap-2.5 px-3.5 border-b border-slate-200 flex-shrink-0">
        <div className="w-6 h-6 bg-blue-600 rounded-md flex items-center justify-center flex-shrink-0">
          <Warehouse size={13} className="text-white" />
        </div>
        {open && (
          <span className="text-sm font-semibold text-slate-900 whitespace-nowrap overflow-hidden">BizAI</span>
        )}
      </div>

      <nav className="flex-1 p-2 space-y-0.5 overflow-hidden">
        {navItems.map(({ id, label, icon: Icon }) => {
          const active = view === id;
          return (
            <button
              key={id}
              onClick={() => setView(id)}
              title={!open ? label : undefined}
              className={`w-full flex items-center gap-2.5 px-2.5 py-2.5 rounded-lg text-xs transition-all duration-150 whitespace-nowrap overflow-hidden ${active ? "bg-blue-50 text-blue-600 border border-blue-200" : "text-slate-500 hover:text-slate-900 hover:bg-slate-50 border border-transparent"}`}
            >
              <Icon size={14} className="flex-shrink-0" />
              {open && <span className="overflow-hidden text-ellipsis">{label}</span>}
            </button>
          );
        })}
      </nav>

      {open && (
        <div className="p-3 border-t border-slate-200 flex-shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-full bg-blue-50 flex items-center justify-center text-[10px] font-bold text-blue-600 flex-shrink-0">
              {(user?.displayName || user?.username || "?").trim().charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-slate-900 truncate">{user?.displayName || "Chưa đăng nhập"}</p>
              <p className="text-[10px] text-slate-500 font-mono truncate">
                {user ? (user.role === "admin" ? "Quản trị viên" : "Store Manager") : ""}
              </p>
            </div>
            <button
              onClick={onLogout}
              className="text-slate-400 hover:text-red-500 p-1 rounded transition-colors"
              title="Đăng xuất"
            >
              <LogOut size={13} />
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

/* ── App ──────────────────────────────────────────────────────── */
// apiBase mặc định theo ngữ cảnh trang:
// - chạy qua nginx docker (:8501 hoặc origin khác) -> dùng cùng origin (nginx đã proxy /api)
// - chạy dev server Vite (5173/4173/3000) -> gọi thẳng backend localhost:8000
const DEV_PORTS = ["5173", "4173", "3000"];
function defaultApiBase(): string {
  try {
    if (typeof window !== "undefined" && window.location?.port
        && !DEV_PORTS.includes(window.location.port)) {
      return window.location.origin;
    }
  } catch { /* bỏ qua - fallback bên dưới */ }
  return "http://localhost:8000";
}

export default function App() {
  const [view, setView] = useState<View>("sales");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [branchId, setBranchId] = useState("all");
  const [apiBase, setApiBase] = useState(defaultApiBase);
  const [liveStores, setLiveStores] = useState<number[] | null>(null);

  // ── Auth state (Row-Level Isolation) ──
  const [auth, setAuth] = useState<AuthInfo | null>(() => restoreAuth());

  function handleLoginSuccess(a: AuthInfo) {
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(a));
    setAuth(a);
    setBranchId("all");
    setLiveStores(null);
  }

  function handleLogout() {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuth(null);
    setLiveStores(null);
    setBranchId("all");
  }

  // Token hết hạn / bị từ chối ở bất kỳ call site nào -> quay về màn hình đăng nhập
  const handleAuthError = useCallback(() => {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuth(null);
    setLiveStores(null);
    setBranchId("all");
  }, []);

  useEffect(() => {
    if (!auth) return;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout>;

    function fetchStores() {
      apiFetch(apiBase, "/api/stores", auth)
        .then((res) => {
          if (res.status === 401) { handleAuthError(); return Promise.reject(); }
          return res.ok ? res.json() : Promise.reject();
        })
        .then((data: number[]) => { if (!cancelled) setLiveStores(data); })
        .catch(() => {
          if (!cancelled) {
            setLiveStores(null);
            retryTimer = setTimeout(fetchStores, 3000);
          }
        });
    }

    fetchStores();
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
    };
  }, [apiBase, auth]);

  const branches = liveStores
    ? [
      { id: "all", label: "Tất cả cơ sở", city: `${liveStores.length} cửa hàng` },
      ...liveStores.map((s) => ({ id: String(s), label: `Cửa hàng #${s}`, city: "" })),
    ]
    : [{ id: "all", label: "Đang kết nối backend...", city: "" }];

  if (!auth) {
    return <LoginView apiBase={apiBase} setApiBase={setApiBase} onLoggedIn={handleLoginSuccess} />;
  }

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden" style={{ fontFamily: "'Be Vietnam Pro', system-ui, sans-serif" }}>
      <Sidebar
        view={view}
        setView={setView}
        open={sidebarOpen}
        user={{ displayName: auth.displayName, role: auth.role }}
        onLogout={handleLogout}
      />

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* TopBar */}
        <header className="h-12 flex-shrink-0 border-b border-slate-200 bg-white flex items-center px-4 gap-3">
          <button
            onClick={() => setSidebarOpen((x) => !x)}
            className="text-slate-500 hover:text-slate-900 transition-colors p-1 rounded-md hover:bg-slate-100"
            title={sidebarOpen ? "Thu gọn menu" : "Mở rộng menu"}
          >
            <Menu size={15} />
          </button>
          <div className="relative max-w-xs w-full hidden sm:block">
            <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500/40 transition-colors"
              placeholder="Tìm kiếm toàn hệ thống..."
            />
          </div>
          <div className="ml-auto flex items-center gap-2">
            <input
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              placeholder="http://localhost:8000"
              className="hidden md:block text-[10px] px-2.5 py-1.5 rounded-lg bg-slate-50 border border-slate-200 w-44 font-mono outline-none focus:border-blue-500/40 transition-colors"
              title="Địa chỉ backend API"
            />
            <div className={`hidden lg:flex items-center gap-1.5 text-[10px] font-mono rounded-full px-2.5 py-1 border ${liveStores !== null ? "text-emerald-500 bg-emerald-500/10 border-emerald-500/20" : "text-red-500 bg-red-500/10 border-red-500/25"}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${liveStores !== null ? "bg-emerald-500" : "bg-red-500"}`} />
              {liveStores !== null ? "Hệ thống hoạt động" : "Mất kết nối backend"}
            </div>
            <button className="relative w-8 h-8 flex items-center justify-center rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors">
              <Bell size={14} />
              <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-red-500 rounded-full" />
            </button>
          </div>
        </header>

        {/* Branch tab bar */}
        <BranchBar branches={branches} branchId={branchId} onSelect={setBranchId} />

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-5">
          {view === "sales" && <SalesForecast branchId={branchId} apiBase={apiBase} auth={auth} onAuthError={handleAuthError} />}
          {view === "chatbot" && <AIChatbot apiBase={apiBase} auth={auth} onAuthError={handleAuthError} />}
          {view === "analysis" && <ProductAnalysis branchId={branchId} apiBase={apiBase} auth={auth} onAuthError={handleAuthError} />}
          {view === "products" && <ProductManagement branchId={branchId} apiBase={apiBase} auth={auth} onAuthError={handleAuthError} />}
          {view === "scenario" && <ScenarioLab branchId={branchId} apiBase={apiBase} auth={auth} onAuthError={handleAuthError} />}
        </main>
      </div>
    </div>
  );
}
