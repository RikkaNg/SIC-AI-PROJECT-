import { useState, useRef, useEffect, useCallback } from "react";
import type { ReactNode, ElementType } from "react";
import {
  AreaChart, Area, ComposedChart, Line,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import {
  TrendingUp, Bot, BarChart2, Package, Warehouse,
  Send, ChevronLeft, ChevronRight, ChevronUp, ChevronDown, Search,
  Bell, Menu, LogOut,
  MapPin, RefreshCw,
  ArrowUpRight, ArrowDownRight, Building2, PackagePlus,
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
  family: string;
  class_code: number | null;
  perishable: number;
  unit_sales: number;
  share_pct: number;
}

interface PieDatum {
  name: string;
  value: number;
}

/** Dòng danh mục sản phẩm thật từ /api/products */
interface ProductItem {
  item_nbr: number;
  family: string;
  class_code: number | null;
  perishable: number;
  stock: number;
  sold_2016: number;
  status: "active" | "outofstock";
}

interface ProductsResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  low_stock_count: number;
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
  const [liveKpi, setLiveKpi] = useState<{ total: string; avgPerDay: string } | null>(null);
  const [topProducts, setTopProducts] = useState<TopProduct[]>([]);
  const [topErr, setTopErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [apiOk, setApiOk] = useState(false);
  const [apiErr, setApiErr] = useState("");

  const isRealStore = branchId === "all" || /^\d+$/.test(branchId);

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
        const storeParam = branchId !== "all" ? `?store_nbr=${branchId}` : "";
        const kpiParam = branchId !== "all" ? `?store_nbr=${branchId}` : "";
        const [predsRes, kpiRes] = await Promise.all([
          apiFetch(apiBase, `/api/predictions${storeParam}`, auth),
          apiFetch(apiBase, `/api/kpi${kpiParam}`, auth),
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
  }, [branchId, apiBase, auth, isRealStore, onAuthError]);

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

  return (
    <div className="space-y-5">
      <SectionHeader title="Dự báo doanh số" sub="Phân tích & dự báo doanh thu theo thời gian thực — Nguồn: model LightGBM thật" />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {liveKpi && (
          <>
            <StatCard label="Tổng dự báo (toàn kỳ)" value={liveKpi.total} sub="Đơn vị: số lượng bán" trend="up" color="blue" />
            <StatCard label="Trung bình mỗi ngày" value={liveKpi.avgPerDay} sub="Gộp tất cả ngành hàng" trend="up" color="green" />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 bg-white border border-slate-200 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-900">Dự báo doanh số theo ngày</p>
              <p className="text-[10px] font-mono text-slate-500 mt-0.5">Tổng tất cả ngành hàng</p>
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
                      <span className="font-mono font-medium">#{p.item_nbr}</span>
                      <span className="text-[10px] font-mono text-slate-400 truncate">{p.family}</span>
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
      // Gửi kèm tối đa 16 lượt gần nhất để AI nhớ ngữ cảnh mà không quá tải token
      const response = await apiFetch(apiBase, "/api/chat", auth, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_query: text, chat_history: historyRef.current.slice(-16) })
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
  const [pieData, setPieData] = useState<PieDatum[]>([]);
  const [totalForecast, setTotalForecast] = useState<number | null>(null);
  const [skuCount, setSkuCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setErr("");
      try {
        const storeParam = branchId !== "all" ? `&store_nbr=${branchId}` : "";
        const [trendRes, mixRes, prodRes] = await Promise.all([
          apiFetch(apiBase, `/api/family-trend?days=${trendDays}&top_families=6${storeParam}`, auth),
          apiFetch(apiBase, `/api/family-mix?top=4${storeParam}`, auth),
          apiFetch(apiBase, `/api/products?page=1&page_size=1${storeParam}`, auth),
        ]);
        if (trendRes.status === 401 || mixRes.status === 401 || prodRes.status === 401) { onAuthError(); return; }
        if (!trendRes.ok) {
          const d = await trendRes.json().catch(() => null);
          throw new Error(typeof d?.detail === "string" ? d.detail : "Không tải được xu hướng dự báo.");
        }
        const trend = await trendRes.json();
        const mix = mixRes.ok ? await mixRes.json() : { items: [] };
        const prod = prodRes.ok ? await prodRes.json() : null;
        if (cancelled) return;

        setTrendRows(trend.series || []);
        setFamilies(trend.families || []);
        setHorizon({ from: trend.date_from, to: trend.date_to });
        setPieData(mix.items || []);
        setTotalForecast((trend.series || []).reduce((s: number, r: any) => {
          return s + (trend.families || []).reduce((s2: number, f: string) => s2 + (Number(r[f]) || 0), 0);
        }, 0));
        setSkuCount(prod?.total ?? null);
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || "Không kết nối được backend.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [branchId, apiBase, auth, onAuthError, trendDays]);

  // ── Dữ liệu bán hàng LỊCH SỬ (dataset tổng toàn chuỗi trong public/) ──
  const [histRows, setHistRows] = useState<RawSaleData[]>([]);
  useEffect(() => {
    let cancelled = false;
    Papa.parse<RawSaleData>("/sale_dataset_ver_2.csv", {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (result) => {
        if (!cancelled) setHistRows(result.data.filter((d) => d.date && d.family && d.unit_sales != null));
      },
      error: () => { /* CSV thiếu -> các mục lịch sử hiện trạng thái trống */ },
    });
    return () => { cancelled = true; };
  }, []);

  // ── Bộ lọc thời gian phần lịch sử ──
  const [granularity, setGranularity] = useState<HistGranularity>("month");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [specificDate, setSpecificDate] = useState("");

  const bounds = useMemo(() => {
    let min = ""; let max = "";
    for (const r of histRows) {
      if (!min || r.date < min) min = r.date;
      if (!max || r.date > max) max = r.date;
    }
    return { min, max };
  }, [histRows]);

  const range = useMemo(() => {
    let from = fromDate || bounds.min;
    let to = toDate || bounds.max;
    if (from && to && from > to) [from, to] = [to, from];
    return { from, to };
  }, [fromDate, toDate, bounds]);

  const dailyTotals = useMemo(() => {
    const m: Record<string, number> = {};
    for (const r of histRows) m[r.date] = (m[r.date] || 0) + (Number(r.unit_sales) || 0);
    return m;
  }, [histRows]);
  const allDates = useMemo(() => Object.keys(dailyTotals).sort(), [dailyTotals]);

  const agg = useMemo(() => {
    const map = new Map<string, { total: number; perishable: number; durable: number; fams: Record<string, number> }>();
    for (const r of histRows) {
      if (r.date < range.from || r.date > range.to) continue;
      const key = histPeriodKey(r.date, granularity);
      let bucket = map.get(key);
      if (!bucket) { bucket = { total: 0, perishable: 0, durable: 0, fams: {} }; map.set(key, bucket); }
      const v = Number(r.unit_sales) || 0;
      bucket.total += v;
      if (r.perishable === 1) bucket.perishable += v; else bucket.durable += v;
      bucket.fams[r.family] = (bucket.fams[r.family] || 0) + v;
    }
    const series = Array.from(map.entries())
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([period, b]) => ({ period, label: histPeriodLabel(period, granularity), ...b, growth: null as number | null }));
    for (let i = 1; i < series.length; i++) {
      const prev = series[i - 1].total;
      series[i].growth = prev > 0 ? ((series[i].total - prev) / prev) * 100 : null;
    }
    const famTotals: Record<string, number> = {};
    for (const s of series) for (const [fam, v] of Object.entries(s.fams)) famTotals[fam] = (famTotals[fam] || 0) + v;
    const topFamilies = Object.entries(famTotals).sort(([, a], [, b]) => b - a);
    return { series, topFamilies };
  }, [histRows, range.from, range.to, granularity]);

  const trendChartData = useMemo(() => agg.series.map((s) => ({
    label: s.label,
    total: Math.round(s.total),
    perishable: Math.round(s.perishable),
    durable: Math.round(s.durable),
    growth: s.growth === null ? null : parseFloat(s.growth.toFixed(1)),
  })), [agg]);

  const top10Bars = useMemo(() =>
    agg.topFamilies.slice(0, 10).map(([name, value]) => ({ name, value: Math.round(value) })),
  [agg]);

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
      if (d < range.from || d > range.to) continue;
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
  }, [agg, allDates, dailyTotals, range.from, range.to]);

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
  const granLabel = granularity === "day" ? "ngày" : granularity === "month" ? "tháng" : "quý";

  return (
    <div className="space-y-5">
      <SectionHeader title="Phân tích sản phẩm" sub="Hiệu suất danh mục — Dự báo LightGBM (backend) + Doanh số thực tế 2016–2017 (dataset)" />

      {/* KPI */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          label={`Tổng dự báo ${horizon ? `(${horizon.from.slice(5)} → ${horizon.to.slice(5)})` : ""}`}
          value={totalForecast != null ? fmt(totalForecast) : "—"}
          sub="Đơn vị số lượng bán" trend="up" color="blue"
        />
        <StatCard
          label="Số sản phẩm trong danh mục"
          value={skuCount != null ? skuCount.toLocaleString("vi-VN") : "—"}
          sub="SKU đang quản lý" trend="up" color="green"
        />
        <StatCard
          label="Doanh số thực tế (kỳ chọn)"
          value={histRows.length ? fmt(histTotal) : "—"}
          sub={`${agg.series.length} ${granLabel} trong khoảng`} trend="up" color="purple"
        />
        <StatCard
          label="Trung bình mỗi kỳ (thực tế)"
          value={histRows.length && agg.series.length ? fmt(histTotal / agg.series.length) : "—"}
          sub="Toàn bộ ngành hàng" trend="up" color="amber"
        />
      </div>

      {/* Xu hướng dự báo theo danh mục (backend) + chọn cửa sổ dự báo */}
      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <p className="text-sm font-semibold text-slate-900">Xu hướng dự báo theo danh mục</p>
            <p className="text-[10px] font-mono text-slate-500 mt-0.5">
              Đơn vị: số lượng bán / ngày — model LightGBM{horizon ? `, kỳ ${horizon.from} → ${horizon.to}` : ""}
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

      {/* Bộ điều khiển thời gian phần doanh số thực tế */}
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

      {/* Hàng biểu đồ lịch sử 1: tổng theo thời gian + top 10 nhóm hàng */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div className="xl:col-span-3 bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Doanh số thực tế theo thời gian</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-4">
            Thang {granLabel} · {range.from ? `${formatDateVN(range.from)} → ${formatDateVN(range.to)}` : "—"} · nguồn dataset bán hàng (toàn chuỗi)
          </p>
          {!bounds.min ? (
            <div className="h-[260px] flex items-center justify-center text-slate-400 text-xs">Đang tải dataset bán hàng...</div>
          ) : trendChartData.length === 0 ? (
            <div className="h-[260px] flex items-center justify-center text-slate-400 text-xs">Không có dữ liệu trong khoảng đã chọn.</div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={trendChartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradHistTotal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.22} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="label" interval="preserveStartEnd"
                  tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false}
                  tickFormatter={(v) => v >= 1000 ? `${v / 1000}B` : `${v}`} />
                <Tooltip content={<ChartTip />} />
                <Area type="monotone" dataKey="total" name="Doanh số" stroke="#3b82f6" fill="url(#gradHistTotal)" strokeWidth={2} />
              </AreaChart>
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

      {/* Hàng biểu đồ lịch sử 2: tăng trưởng theo kỳ + dễ hỏng vs bền */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Tổng doanh số & mức thay đổi</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-4">Cột: tổng mỗi kỳ — Đường: % thay đổi so với kỳ trước</p>
          {trendChartData.length < 2 ? (
            <div className="h-[240px] flex items-center justify-center text-slate-400 text-xs">Cần ít nhất 2 kỳ để tính mức thay đổi.</div>
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={trendChartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="label" interval="preserveStartEnd"
                  tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <YAxis yAxisId="l" tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false}
                  tickFormatter={(v) => v >= 1000 ? `${v / 1000}B` : `${v}`} />
                <YAxis yAxisId="r" orientation="right" tickFormatter={(v) => `${v}%`}
                  tick={{ fill: "#a855f7", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTip />} />
                <Legend wrapperStyle={{ fontSize: "10px", color: "#64748b", fontFamily: "JetBrains Mono", paddingTop: 8 }} />
                <Bar yAxisId="l" dataKey="total" name="Tổng doanh số" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                <Line yAxisId="r" type="monotone" dataKey="growth" name="% thay đổi" stroke="#a855f7" strokeWidth={2}
                  dot={{ r: 2, fill: "#a855f7", strokeWidth: 0 }} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Hàng dễ hỏng vs hàng bền</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-4">Nhìn nhận mùa vụ của hàng tươi (PRODUCE, DAIRY...)</p>
          {trendChartData.length === 0 ? (
            <div className="h-[240px] flex items-center justify-center text-slate-400 text-xs">Không có dữ liệu.</div>
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={trendChartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="label" interval="preserveStartEnd"
                  tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false}
                  tickFormatter={(v) => v >= 1000 ? `${v / 1000}B` : `${v}`} />
                <Tooltip content={<ChartTip />} />
                <Legend wrapperStyle={{ fontSize: "10px", color: "#64748b", fontFamily: "JetBrains Mono", paddingTop: 8 }} />
                <Line type="monotone" dataKey="perishable" name="Dễ hỏng" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="durable" name="Hàng bền" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
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

      {/* Thị phần danh mục (giữ nguyên pie) */}
      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <p className="text-sm font-semibold text-slate-900">Thị phần danh mục</p>
        <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-3">Tổng unit sales năm 2016 theo nhóm hàng{branchId !== "all" ? ` — Cửa hàng #${branchId}` : ""}</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={38} outerRadius={70} paddingAngle={3} dataKey="value">
                {pieData.map((_, i) => (<Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />))}
              </Pie>
              <Tooltip content={<ChartTip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5">
            {pieData.length === 0 ? (
              <p className="text-slate-500 text-xs">Chưa có dữ liệu thị phần.</p>
            ) : pieData.map((d, i) => (
              <div key={d.name} className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                  <span className="text-[11px] text-slate-700">{d.name}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">{d.value.toLocaleString("vi-VN")}</span>
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
            placeholder="Tìm theo mã sản phẩm, nhóm hàng..."
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
                <th className="text-left px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Nhóm hàng</th>
                <th className="text-left px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Class</th>
                <th className="text-center px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Dễ hỏng</th>
                <th className="text-right px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                  <button className="flex items-center gap-1 ml-auto hover:text-slate-900 transition-colors uppercase" onClick={() => toggleSort("stock")}>Tồn kho <SortIcon k="stock" /></button>
                </th>
                <th className="text-right px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                  <button className="flex items-center gap-1 ml-auto hover:text-slate-900 transition-colors uppercase" onClick={() => toggleSort("sold")}>Bán 2016 <SortIcon k="sold" /></button>
                </th>
                <th className="text-center px-4 py-3 text-[10px] font-mono text-slate-500 uppercase tracking-wider">Trạng thái</th>
                {branchId !== "all" && <th className="px-4 py-3" />}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={branchId !== "all" ? 8 : 7} className="text-center py-8 text-slate-500 text-sm">
                  <RefreshCw size={14} className="animate-spin inline mr-2" />Đang tải...
                </td></tr>
              ) : products.length === 0 ? (
                <tr><td colSpan={branchId !== "all" ? 8 : 7} className="text-center py-8 text-slate-500 text-sm">Không có sản phẩm nào khớp bộ lọc.</td></tr>
              ) : products.map((p) => (
                <tr key={p.item_nbr} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3"><span className="text-xs font-mono font-medium text-slate-900">#{p.item_nbr}</span></td>
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
                    <span className={`text-xs font-mono ${p.stock === 0 ? "text-red-500" : p.stock < 30 * (branchId !== "all" ? 1 : 20) ? "text-orange-500" : "text-emerald-500"}`}>
                      {Math.round(p.stock).toLocaleString("vi-VN")}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-xs font-mono text-slate-700">{Math.round(p.sold_2016).toLocaleString("vi-VN")}</span>
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
                <p className="text-sm font-semibold text-slate-900">Nhập hàng — #{restockTarget.item_nbr}</p>
                <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                  Cửa hàng #{branchId} · Tồn kho hiện tại: {Math.round(restockTarget.stock).toLocaleString("vi-VN")}
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
type View = "sales" | "chatbot" | "analysis" | "products";

const navItems: { id: View; label: string; icon: ElementType }[] = [
  { id: "sales", label: "Dự báo doanh số", icon: TrendingUp },
  { id: "chatbot", label: "AI Chatbot", icon: Bot },
  { id: "analysis", label: "Phân tích sản phẩm", icon: BarChart2 },
  { id: "products", label: "Quản lý sản phẩm", icon: Package },
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

/* ── Shell components (kiểu frontend2) ───────────────────────── */
function Sidebar({
  view, setView, open, user, onLogout,
}: {
  view: View; setView: (v: View) => void; open: boolean;
  user: { displayName: string; role: string } | null;
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
export default function App() {
  const [view, setView] = useState<View>("sales");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [branchId, setBranchId] = useState("all");
  const [apiBase, setApiBase] = useState("http://localhost:8000");
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
        </main>
      </div>
    </div>
  );
}
