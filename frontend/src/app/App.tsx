import { useState, useRef, useEffect } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import {
  TrendingUp, Bot, BarChart2, Package,
  Send, ChevronUp, ChevronDown, Search,
  Bell, Settings, Menu, X, LogOut, User,
  ShoppingCart, DollarSign, MapPin,
  Plus, Edit2, Trash2, Star, RefreshCw,
  Zap, ArrowUpRight, ArrowDownRight, Circle,
  ChevronRight, Building2,
} from "lucide-react";
import Papa from 'papaparse';

/* ── Types ─────────────────────────────────────────────────────── */
interface RawSaleData {
  date: string;
  family: string;
  perishable: number;
  unit_sales: number;
}

interface TopProduct {
  name: string;
  value: number;
  change: number;
}

interface PieDatum {
  name: string;
  value: number;
}

interface TrendDatum {
  week: string;
  grocery: number;
  cleaning: number;
  produce: number;
  beverages: number;
  dairy: number;
  personal: number;
}

interface ProductItem {
  id: number;
  name: string;
  sku: string;
  category: string;
  price: number;
  stock: number;
  status: string;
  rating: number;
  perishable: number;
}

interface Message {
  role: string;
  text: string;
}

/* ── Static Config & Mocks ────────────────────────────────────── */
const PIE_COLORS = ["#06c9f0", "#7c3aed", "#10b981", "#f59e0b", "#6b82a4"];

const trendData: TrendDatum[] = [
  { week: "W1", grocery: 3200, cleaning: 2100, produce: 1400, beverages: 1800, dairy: 1500, personal: 900 },
  { week: "W2", grocery: 2900, cleaning: 2400, produce: 1600, beverages: 1900, dairy: 1600, personal: 850 },
  { week: "W3", grocery: 3800, cleaning: 2600, produce: 1300, beverages: 2100, dairy: 1700, personal: 920 },
  { week: "W4", grocery: 4200, cleaning: 2900, produce: 1800, beverages: 2200, dairy: 1800, personal: 1050 },
  { week: "W5", grocery: 3600, cleaning: 3100, produce: 2000, beverages: 2000, dairy: 1900, personal: 1100 },
  { week: "W6", grocery: 4500, cleaning: 2800, produce: 1750, beverages: 2300, dairy: 1750, personal: 980 },
];

const initialProducts: ProductItem[] = [
  { id: 1, name: "Gạo thơm đặc biệt", sku: "CLS-1093", category: "GROCERY I", price: 0, stock: 1420, status: "active", rating: 4.8, perishable: 0 },
  { id: 2, name: "Nước xả vải", sku: "CLS-3008", category: "CLEANING", price: 0, stock: 980, status: "active", rating: 4.6, perishable: 0 },
  { id: 3, name: "Dầu ăn đậu nành", sku: "CLS-1028", category: "GROCERY I", price: 0, stock: 760, status: "active", rating: 4.9, perishable: 0 },
  { id: 4, name: "Rau xà lách tươi", sku: "CLS-1122", category: "PRODUCE", price: 0, stock: 18, status: "outofstock", rating: 4.7, perishable: 1 },
  { id: 5, name: "Sữa tươi nguyên kem", sku: "CLS-2011", category: "DAIRY", price: 0, stock: 203, status: "active", rating: 4.5, perishable: 1 },
  { id: 6, name: "Nước suối đóng chai", sku: "CLS-4050", category: "BEVERAGES", price: 0, stock: 0, status: "outofstock", rating: 4.8, perishable: 0 },
  { id: 7, name: "Kem đánh răng", sku: "CLS-5021", category: "PERSONAL CARE", price: 0, stock: 31, status: "active", rating: 4.4, perishable: 0 },
  { id: 8, name: "Thịt nguội đóng gói", sku: "CLS-6032", category: "DELI", price: 0, stock: 94, status: "discontinued", rating: 4.2, perishable: 1 },
];

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

function KpiCard({ icon: Icon, label, value, sub, color, up }: {
  icon: any; label: string; value: string; sub: string; color: string; up?: boolean;
}) {
  return (
    <div className="bg-card border border-border rounded-xl p-5 flex gap-4 items-start hover:border-primary/30 transition-colors">
      <div className={`w-11 h-11 rounded-lg flex items-center justify-center shrink-0 ${color}`}>
        <Icon size={20} />
      </div>
      <div className="min-w-0">
        <p className="text-muted-foreground text-xs font-medium uppercase tracking-wider mb-1">{label}</p>
        <p className="text-foreground text-xl font-bold font-mono leading-none">{value}</p>
        <p className={`text-xs mt-1 flex items-center gap-1 ${up === undefined ? "text-muted-foreground" : up ? "text-emerald-400" : "text-rose-400"}`}>
          {up !== undefined && (up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />)}
          {sub}
        </p>
      </div>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d1528] border border-border rounded-lg p-3 text-xs shadow-xl">
      <p className="text-muted-foreground mb-2 font-mono">{label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-foreground/70">{p.name}:</span>
          <span className="text-foreground font-mono font-medium">{p.value.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
};

/* ── Section: Dự báo doanh số ─────────────────────────────────── */
function SalesForecast({ branchId, apiBase, productSales }: { branchId: string; apiBase: string; productSales: TopProduct[] }) {
  const [liveChart, setLiveChart] = useState<Array<{ date: string; dubao: number }> | null>(null);
  const [liveKpi, setLiveKpi] = useState<{ total: string; avgPerDay: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [apiOk, setApiOk] = useState(false);

  const isRealStore = branchId === "all" || /^\d+$/.test(branchId);

  useEffect(() => {
    if (!isRealStore) {
      setApiOk(false);
      setLoading(false);
      return;
    }

    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const storeParam = branchId !== "all" ? `?store_nbr=${branchId}` : "";
        const kpiParam = branchId !== "all" ? `?store_nbr=${branchId}` : "";
        const [predsRes, kpiRes] = await Promise.all([
          fetch(`${apiBase}/api/predictions${storeParam}`, { headers: { "ngrok-skip-browser-warning": "true" } }),
          fetch(`${apiBase}/api/kpi${kpiParam}`, { headers: { "ngrok-skip-browser-warning": "true" } }),
        ]);
        if (!predsRes.ok || !kpiRes.ok) throw new Error("API lỗi");
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
      } catch (e) {
        if (!cancelled) setApiOk(false);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [branchId, apiBase, isRealStore]);

  const chartData = liveChart ?? [];

  if (!isRealStore || loading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm gap-2">
        <RefreshCw size={15} className="animate-spin" />
        Đang tải dữ liệu từ backend...
      </div>
    );
  }

  if (!apiOk) {
    return (
      <div className="bg-rose-500/10 border border-rose-500/25 rounded-lg px-4 py-3 text-sm text-rose-300">
        Không kết nối được backend ({apiBase}). Kiểm tra lại backend đã chạy chưa.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {liveKpi && (
          <>
            <KpiCard icon={DollarSign} label="Tổng dự báo (toàn kỳ)" value={liveKpi.total} sub="Đơn vị: số lượng bán" color="bg-cyan-500/10 text-cyan-400" up />
            <KpiCard icon={TrendingUp} label="Trung bình mỗi ngày" value={liveKpi.avgPerDay} sub="Gộp tất cả ngành hàng" color="bg-violet-500/10 text-violet-400" up />
          </>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-foreground font-semibold">Dự báo doanh số theo ngày</h3>
              <p className="text-muted-foreground text-xs mt-0.5">Nguồn: model LightGBM thật</p>
            </div>
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-violet-400 inline-block rounded" />Dự báo</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="date" tick={{ fill: "#6b82a4", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#6b82a4", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => v >= 1000 ? `${v / 1000}B` : `${v}M`} />
              <Tooltip content={<CustomTooltip />} />
              <Line dataKey="dubao" name="Dự báo" stroke="#7c3aed" strokeWidth={2} dot={{ fill: "#7c3aed", r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-foreground font-semibold mb-4">Top sản phẩm bán chạy</h3>
          <div className="space-y-3">
            {productSales.length === 0 ? (
              <p className="text-muted-foreground text-xs">Đang tải dữ liệu CSV...</p>
            ) : (
              productSales.map((p, i) => (
                <div key={p.name}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-foreground/80 text-xs">{p.name}</span>
                    <span className={`text-xs font-mono flex items-center gap-0.5 ${p.change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      {p.change >= 0 ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
                      {Math.abs(p.change)}%
                    </span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all duration-700"
                      style={{ width: `${(p.value / (productSales[0]?.value || 1)) * 100}%`, background: PIE_COLORS[i] }}
                    />
                  </div>
                  <p className="text-muted-foreground text-xs mt-1 font-mono">{p.value.toLocaleString()} đơn</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Section: AI Chatbot (Tích hợp Groq) ──────────────────────────── */
function AIChatbot({ aiContext }: { aiContext: string }) {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Xin chào! Tôi là AI Assistant dùng Llama 3.1 qua Groq. Hãy hỏi tôi về tình hình kinh doanh theo ngày, tháng, sản phẩm nhé!" }
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  const send = async (text: string) => {
    if (!text.trim() || thinking) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setThinking(true);

    try {
      const promptText = `
        Bạn là một AI Business Assistant chuyên nghiệp.
        Dưới đây là bối cảnh dữ liệu (Context Window) bạn cần nắm vững. Chú ý đặc biệt phần dữ liệu chi tiết theo từng Ngày, tháng, năm:
        ${aiContext || "Đang chờ tải dữ liệu..."}

        Dịch các cột trong file dữ liệu bạn đang đọc:
          store_nbr: Mã số định danh cửa hàng.
          family: là các loại sản phẩm được gộp từ nhiều sản phẩm có cũng công dụng (PRODUCE, BEVERAGES, CLEANING, DAIRY, GROCERY I...).
          date: Ngày ghi nhận dữ liệu giao dịch.
          unit_sales: Tổng số lượng sản phẩm bán ra thực tế trong ngày.
          onpromotion: Cột đánh dấu trạng thái khuyến mãi (1: đang có khuyến mãi, 0: không có).
          oil_price: Giá dầu thô trong ngày.
          is_earthquake_period: Đánh dấu giai đoạn chịu ảnh hưởng bởi trận động đất lớn tại Ecuador năm 2016 (1: động đất, 0: bình thường).
          holiday_type: Phân loại ngày lễ/sự kiện trong ngày.
          city: Thành phố nơi đặt cửa hàng.
          type: Loại quy mô cửa hàng (A đến E).
          transactions_lag1: Tổng số lượt giao dịch của cửa hàng đó ở 1 ngày trước đó.
          target: Biến mục tiêu đưa vào mô hình huấn luyện.
        Mục tiêu của AI:
          - Phân tích tình hình kinh doanh dựa trên dữ liệu lịch sử.
          - Dự báo unit_sales trong tương lai.
          - Xác định các yếu tố có ảnh hưởng đến doanh số.
          - Đưa ra khuyến nghị thực tế cho chủ cửa hàng.
          - Không tự bịa dữ liệu nếu context không cung cấp đủ thông tin.
        Nhiệm vụ: Dựa vào bối cảnh trên, hãy trả lời câu hỏi người dùng ngắn gọn, súc tích, có phân tích và đưa ra lời khuyên. 
        Trả lời tiếng Việt.

        Câu hỏi của người dùng: "${text}"
      `;

      // Gọi API SiliconFlow (Qwen 2.5)
      const response = await fetch("https://api.siliconflow.cn/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer skxxxxxxxxxxx` // Dán Key SiliconFlow của bạn vào đây
        },
        body: JSON.stringify({
          model: "Qwen/Qwen2.5-7B-Instruct", // Model Qwen miễn phí
          messages: [
            { role: "system", content: promptText },
            { role: "user", content: text }
          ],
          temperature: 0.4
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("SiliconFlow Lỗi chi tiết:", errorText);
        throw new Error(`Chi tiết lỗi từ server: ${errorText}`);
      }

      const data = await response.json();
      const aiText = data?.choices?.[0]?.message?.content || "Tôi không hiểu phản hồi từ AI.";
      setMessages((m) => [...m, { role: "assistant", text: aiText }]);
    } catch (error: any) {
      console.error("Lỗi hệ thống:", error);
      setMessages((m) => [...m, { role: "assistant", text: `Lỗi AI: ${error.message}` }]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 h-[calc(100vh-180px)] min-h-[500px]">
      <div className="xl:col-span-3 bg-card border border-border rounded-xl flex flex-col overflow-hidden">
        <div className="p-4 border-b border-border flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center">
            <Bot size={18} className="text-white" />
          </div>
          <div>
            <p className="text-foreground font-semibold text-sm">AI Business Assistant</p>
            <p className="text-emerald-400 text-xs flex items-center gap-1.5">
              <Circle size={6} fill="currentColor" />Powered by Groq Llama 3.1
            </p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
              <div className={`w-7 h-7 rounded-lg shrink-0 flex items-center justify-center text-xs font-bold ${m.role === "assistant" ? "bg-gradient-to-br from-cyan-500 to-violet-600 text-white" : "bg-primary/20 text-primary"}`}>
                {m.role === "assistant" ? <Bot size={14} /> : <User size={14} />}
              </div>
              <div className={`max-w-[80%] rounded-xl px-4 py-3 text-sm leading-relaxed ${m.role === "assistant" ? "bg-muted text-foreground" : "bg-primary/15 text-foreground border border-primary/20"}`}>
                {m.text.split("\n").map((line, j) => (
                  <p key={j} className={line === "" ? "h-2" : ""}>
                    {line.startsWith("**") && line.endsWith("**")
                      ? <strong className="text-primary">{line.slice(2, -2)}</strong>
                      : line}
                  </p>
                ))}
              </div>
            </div>
          ))}
          {thinking && (
            <div className="flex gap-3">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center">
                <Bot size={14} className="text-white" />
              </div>
              <div className="bg-muted rounded-xl px-4 py-3 flex items-center gap-2">
                {[0, 1, 2].map((i) => (
                  <span key={i} className="w-1.5 h-1.5 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="p-4 border-t border-border">
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send(input)}
              placeholder="Hỏi AI về dữ liệu cửa hàng..."
              className="flex-1 bg-muted border border-border rounded-lg px-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50 transition-colors"
            />
            <button
              onClick={() => send(input)}
              disabled={!input.trim() || thinking}
              className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center text-primary-foreground hover:bg-primary/80 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase tracking-wider mb-3">Câu hỏi nhanh</p>
          <div className="space-y-2">
            {quickQuestions.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                className="w-full text-left text-xs text-foreground/80 bg-muted hover:bg-primary/10 hover:text-primary border border-border hover:border-primary/30 rounded-lg px-3 py-2 transition-all flex items-center gap-2"
              >
                <Zap size={11} className="text-primary shrink-0" />
                {q}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase tracking-wider mb-3">Thống kê AI</p>
          <div className="space-y-3">
            {[{ label: "Model đang dùng", val: "Groq Llama 3.1" }, { label: "Thời gian phản hồi", val: "1.0s" }].map((s) => (
              <div key={s.label} className="flex items-center justify-between">
                <span className="text-muted-foreground text-xs">{s.label}</span>
                <span className="text-foreground font-mono text-xs font-semibold">{s.val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Section: Phân tích sản phẩm ─────────────────────────────── */
function ProductAnalysis({ pieData }: { pieData: PieDatum[] }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard icon={ShoppingCart} label="Đơn hàng T6" value="4,320" sub="+12.8% vs T5" color="bg-emerald-500/10 text-emerald-400" up={true} />
        <KpiCard icon={TrendingUp} label="Tỷ lệ chuyển đổi" value="3.82%" sub="-0.3pp vs T5" color="bg-violet-500/10 text-violet-400" up={false} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div className="xl:col-span-3 bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="text-foreground font-semibold">Xu hướng doanh số theo danh mục</h3>
              <p className="text-muted-foreground text-xs mt-0.5">Đơn vị: số lượng bán / tuần</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={trendData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorGrocery" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#06c9f0" stopOpacity={0.3} /><stop offset="95%" stopColor="#06c9f0" stopOpacity={0} /></linearGradient>
                <linearGradient id="colorClean" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#7c3aed" stopOpacity={0.3} /><stop offset="95%" stopColor="#7c3aed" stopOpacity={0} /></linearGradient>
                <linearGradient id="colorProduce" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.3} /><stop offset="95%" stopColor="#10b981" stopOpacity={0} /></linearGradient>
                <linearGradient id="colorBev" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} /><stop offset="95%" stopColor="#f59e0b" stopOpacity={0} /></linearGradient>
                <linearGradient id="colorDairy" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} /><stop offset="95%" stopColor="#ef4444" stopOpacity={0} /></linearGradient>
                <linearGradient id="colorPers" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ec4899" stopOpacity={0.3} /><stop offset="95%" stopColor="#ec4899" stopOpacity={0} /></linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="week" tick={{ fill: "#6b82a4", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#6b82a4", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: "#0d1528", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11, color: "#6b82a4", paddingTop: 10 }} />
              <Area type="monotone" dataKey="grocery" name="GROCERY I" stroke="#06c9f0" fill="url(#colorGrocery)" strokeWidth={2} />
              <Area type="monotone" dataKey="cleaning" name="CLEANING" stroke="#7c3aed" fill="url(#colorClean)" strokeWidth={2} />
              <Area type="monotone" dataKey="produce" name="PRODUCE" stroke="#10b981" fill="url(#colorProduce)" strokeWidth={2} />
              <Area type="monotone" dataKey="beverages" name="BEVERAGES" stroke="#f59e0b" fill="url(#colorBev)" strokeWidth={2} />
              <Area type="monotone" dataKey="dairy" name="DAIRY" stroke="#ef4444" fill="url(#colorDairy)" strokeWidth={2} />
              <Area type="monotone" dataKey="personal" name="PERSONAL CARE" stroke="#ec4899" fill="url(#colorPers)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="xl:col-span-2 bg-card border border-border rounded-xl p-5">
          <h3 className="text-foreground font-semibold mb-5">Thị phần danh mục</h3>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={52} outerRadius={80} paddingAngle={3} dataKey="value">
                {pieData.map((_, i) => (<Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} stroke="transparent" />))}
              </Pie>
              <Tooltip contentStyle={{ background: "#0d1528", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8, fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-2 mt-2">
            {pieData.length === 0 ? <p className="text-muted-foreground text-xs">Đang tải dữ liệu...</p> : pieData.map((d, i) => (
              <div key={d.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                  <span className="text-muted-foreground text-xs">{d.name}</span>
                </div>
                <span className="text-foreground text-xs font-mono font-semibold">{d.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Section: Quản lý sản phẩm ───────────────────────────────── */
function ProductManagement() {
  const [products, setProducts] = useState<ProductItem[]>(initialProducts);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [sortKey, setSortKey] = useState<"name" | "stock" | "rating">("name");
  const [sortAsc, setSortAsc] = useState(true);

  const filtered = products
    .filter((p) => { const q = search.toLowerCase(); return p.name.toLowerCase().includes(q) || p.sku.toLowerCase().includes(q); })
    .filter((p) => filter === "all" || p.status === filter)
    .sort((a, b) => {
      const v = sortAsc ? 1 : -1;
      if (sortKey === "name") return v * a.name.localeCompare(b.name);
      if (sortKey === "rating") return v * (a.rating - b.rating);
      return v * (a.stock - b.stock);
    });

  const toggleSort = (k: typeof sortKey) => {
    if (sortKey === k) setSortAsc((x) => !x);
    else { setSortKey(k); setSortAsc(true); }
  };

  const statusBadge = (s: string) => {
    const map: Record<string, string> = {
      active: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20",
      outofstock: "bg-amber-500/15 text-amber-400 border-amber-500/20",
      discontinued: "bg-rose-500/15 text-rose-400 border-rose-500/20",
    };
    const labels: Record<string, string> = { active: "Đang bán", outofstock: "Hết hàng", discontinued: "Ngừng KD" };
    return <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${map[s]}`}>{labels[s]}</span>;
  };

  const SortIcon = ({ k }: { k: typeof sortKey }) =>
    sortKey === k ? (sortAsc ? <ChevronUp size={12} className="text-primary" /> : <ChevronDown size={12} className="text-primary" />) : <ChevronUp size={12} className="text-muted-foreground/30" />;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-3 items-center">
        <div className="relative flex-1 min-w-[200px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Tìm theo tên, SKU..." className="w-full bg-card border border-border rounded-lg pl-9 pr-4 py-2.5 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50 transition-colors" />
        </div>
        <div className="flex gap-1 bg-card border border-border rounded-lg p-1">
          {[{ key: "all", label: "Tất cả" }, { key: "active", label: "Đang bán" }, { key: "outofstock", label: "Hết hàng" }, { key: "discontinued", label: "Ngừng KD" }].map((f) => (
            <button key={f.key} onClick={() => setFilter(f.key)} className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${filter === f.key ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}>{f.label}</button>
          ))}
        </div>
        <button className="flex items-center gap-2 bg-primary/10 border border-primary/20 hover:bg-primary/20 text-primary rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ml-auto">
          <Plus size={15} />Thêm sản phẩm
        </button>
      </div>

      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left px-5 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  <button className="flex items-center gap-1.5 hover:text-foreground transition-colors" onClick={() => toggleSort("name")}>Sản phẩm <SortIcon k="name" /></button>
                </th>
                <th className="text-left px-4 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">Mã Class (SKU)</th>
                <th className="text-left px-4 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">Nhóm hàng (Family)</th>
                <th className="text-center px-4 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">Dễ hỏng</th>
                <th className="text-right px-4 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  <button className="flex items-center gap-1.5 hover:text-foreground transition-colors ml-auto" onClick={() => toggleSort("stock")}>Tồn kho <SortIcon k="stock" /></button>
                </th>
                <th className="text-center px-4 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  <button className="flex items-center gap-1.5 hover:text-foreground transition-colors mx-auto" onClick={() => toggleSort("rating")}>Đánh giá <SortIcon k="rating" /></button>
                </th>
                <th className="text-center px-4 py-3.5 text-muted-foreground text-xs font-medium uppercase tracking-wider">Trạng thái</th>
                <th className="px-4 py-3.5" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => (
                <tr key={p.id} className={`border-b border-border/60 hover:bg-muted/40 transition-colors ${i === filtered.length - 1 ? "border-b-0" : ""}`}>
                  <td className="px-5 py-3.5"><p className="text-foreground text-sm font-medium">{p.name}</p></td>
                  <td className="px-4 py-3.5"><span className="text-muted-foreground text-xs font-mono">{p.sku}</span></td>
                  <td className="px-4 py-3.5"><span className="text-muted-foreground text-xs">{p.category}</span></td>
                  <td className="px-4 py-3.5 text-center">
                    <span className={`text-xs font-mono px-2 py-0.5 rounded ${p.perishable === 1 ? "bg-rose-500/15 text-rose-400" : "bg-emerald-500/15 text-emerald-400"}`}>{p.perishable === 1 ? "Có" : "Không"}</span>
                  </td>
                  <td className="px-4 py-3.5 text-right"><span className={`text-sm font-mono ${p.stock === 0 ? "text-rose-400" : p.stock < 30 ? "text-amber-400" : "text-foreground"}`}>{p.stock}</span></td>
                  <td className="px-4 py-3.5 text-center"><span className="text-amber-400 text-xs font-mono flex items-center justify-center gap-1"><Star size={11} fill="currentColor" />{p.rating}</span></td>
                  <td className="px-4 py-3.5 text-center">{statusBadge(p.status)}</td>
                  <td className="px-4 py-3.5">
                    <div className="flex items-center gap-1.5 justify-end">
                      <button className="p-1.5 text-muted-foreground hover:text-primary hover:bg-primary/10 rounded transition-colors"><Edit2 size={13} /></button>
                      <button className="p-1.5 text-muted-foreground hover:text-rose-400 hover:bg-rose-500/10 rounded transition-colors" onClick={() => setProducts((pp) => pp.filter((x) => x.id !== p.id))}><Trash2 size={13} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="px-5 py-3 border-t border-border flex items-center justify-between">
          <p className="text-muted-foreground text-xs">Hiển thị {filtered.length} / {products.length} sản phẩm</p>
          <p className="text-muted-foreground text-xs font-mono">{products.filter((p) => p.stock < 30).length} sản phẩm cần bổ sung tồn kho</p>
        </div>
      </div>
    </div>
  );
}

/* ── Nav config ───────────────────────────────────────────────── */
const navItems = [
  { id: "sales", label: "Dự báo doanh số", icon: TrendingUp },
  { id: "chatbot", label: "AI Chatbot", icon: Bot },
  { id: "analysis", label: "Phân tích sản phẩm", icon: BarChart2 },
  { id: "products", label: "Quản lý sản phẩm", icon: Package },
];

const sectionTitles: Record<string, { title: string; sub: string }> = {
  sales: { title: "Dự báo doanh số", sub: "Phân tích & dự báo doanh thu theo thời gian thực" },
  chatbot: { title: "AI Chatbot", sub: "Trợ lý AI thông minh phân tích kinh doanh" },
  analysis: { title: "Phân tích sản phẩm", sub: "Hiệu suất và xu hướng danh mục sản phẩm" },
  products: { title: "Quản lý sản phẩm", sub: "Toàn bộ danh mục sản phẩm & tồn kho" },
};

/* ── App ──────────────────────────────────────────────────────── */
export default function App() {
  const [active, setActive] = useState("sales");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [branchId, setBranchId] = useState("all");
  const [apiBase, setApiBase] = useState("http://localhost:8000");
  const [liveStores, setLiveStores] = useState<number[] | null>(null);

  // State cho dữ liệu CSV và AI Context
  const [topProducts, setTopProducts] = useState<TopProduct[]>([]);
  const [pieData, setPieData] = useState<PieDatum[]>([]);
  const [aiContext, setAiContext] = useState<string>("");

  // Load dữ liệu từ file CSV đặt trong thư mục public
  useEffect(() => {
    Papa.parse<RawSaleData>('/sale_dataset_ver_2.csv', {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (result) => {
        const data = result.data.filter(d => d.family && d.unit_sales != null);

        // 1. Gom nhóm theo family và tính tổng unit_sales
        const salesByFamily: Record<string, number> = {};
        for (const item of data) {
          salesByFamily[item.family] = (salesByFamily[item.family] || 0) + item.unit_sales;
        }

        // 2. Lấy Top 5 sản phẩm bán chạy nhất
        const top5 = Object.entries(salesByFamily)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .map(([name, value]) => ({
            name,
            value: Math.round(value),
            change: parseFloat(((Math.random() * 30 - 10)).toFixed(1))
          }));
        setTopProducts(top5);

        // 3. Tính dữ liệu Pie Chart (Top 4 + "Khác")
        const sortedFamilies = Object.entries(salesByFamily).sort(([, a], [, b]) => b - a);
        const top4 = sortedFamilies.slice(0, 4).map(([name, value]) => ({ name, value: Math.round(value) }));
        const othersValue = sortedFamilies.slice(4).reduce((sum, [, v]) => sum + v, 0);
        if (othersValue > 0) {
          top4.push({ name: "Khác", value: Math.round(othersValue) });
        }
        setPieData(top4);

        // 4. Tính dữ liệu chi tiết theo Ngày và Tháng cho AI
        const monthlySummary: Record<string, Record<string, number>> = {};
        const dailySummary: Record<string, number> = {};

        for (const item of data) {
          const month = item.date.slice(0, 7);
          if (!monthlySummary[month]) monthlySummary[month] = {};
          if (!monthlySummary[month][item.family]) monthlySummary[month][item.family] = 0;
          monthlySummary[month][item.family] += item.unit_sales;

          dailySummary[item.date] = (dailySummary[item.date] || 0) + item.unit_sales;
        }

        let monthContext = "";
        for (const [month, families] of Object.entries(monthlySummary)) {
          monthContext += `Tháng ${month.slice(5)}/${month.slice(0, 4)}:\n`;
          const topFam = Object.entries(families).sort((a, b) => b[1] - a[1]).slice(0, 3);
          for (const [fam, val] of topFam) {
            monthContext += `- ${fam}: ${Math.round(val)} đơn\n`;
          }
        }

        let dailyContext = "";
        for (const [dateStr, total] of Object.entries(dailySummary)) {
          const [year, month, day] = dateStr.split('-');
          dailyContext += `- Ngày ${parseInt(day)} tháng ${parseInt(month)} năm ${year}: ${Math.round(total)} đơn\n`;
        }

        // 5. TẠO CONTEXT WINDOW DÀNH CHO AI
        const totalSales = data.reduce((sum, item) => sum + item.unit_sales, 0);
        const perishableSales = data.filter(d => d.perishable === 1).reduce((sum, item) => sum + item.unit_sales, 0);

        const contextWindow = `
          BỐI CẢNH DỮ LIỆU CỬA HÀNG (Năm 2016):
          1. TỔNG QUAN: Tổng số lượng bán ra trong năm là ${Math.round(totalSales)} đơn vị. Tỷ lệ hàng dễ hỏng (perishable) chiếm ${((perishableSales / totalSales) * 100).toFixed(1)}% tổng doanh số.
          2. TOP 5 NHÓM HÀNG BÁN CHẠY NHẤT CẢ NĂM: 
          ${top5.map(p => `- ${p.name}: ${p.value} đơn`).join('\n')}
          3. PHÂN LOẠI THỊ PHẦN: 
          ${top4.map(p => `- ${p.name}: ${p.value} đơn`).join('\n')}
          4. CHI TIẾT DOANH SỐ THEO TỪNG THÁNG (Top 3 nhóm hàng mỗi tháng):
          ${monthContext}
          5. CHI TIẾT DOANH SỐ TỔNG THEO TỪNG NGÀY (Dữ liệu rất quan trọng, đọc kỹ ngày, tháng, năm):
          ${dailyContext}
        `;
        setAiContext(contextWindow);
      },
      error: (error) => {
        console.error("Lỗi đọc file CSV:", error);
      }
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout>;

    function fetchStores() {
      fetch(`${apiBase}/api/stores`, { headers: { "ngrok-skip-browser-warning": "true" } })
        .then((res) => (res.ok ? res.json() : Promise.reject()))
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
  }, [apiBase]);

  const branches = liveStores
    ? [
      { id: "all", label: "Tất cả cơ sở", city: `${liveStores.length} cửa hàng` },
      ...liveStores.map((s) => ({ id: String(s), label: `Cửa hàng #${s}`, city: "" })),
    ]
    : [{ id: "all", label: "Đang kết nối backend...", city: "" }];

  return (
    <div className="flex h-screen bg-background overflow-hidden" style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif" }}>
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? "w-60" : "w-16"} shrink-0 bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300 overflow-hidden`}>
        <div className="p-4 border-b border-sidebar-border flex items-center gap-3 min-h-[60px]">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-violet-600 flex items-center justify-center shrink-0">
            <Zap size={16} className="text-white" />
          </div>
          {sidebarOpen && (
            <div className="overflow-hidden">
              <p className="text-foreground font-bold text-sm leading-none">BizAI</p>
              <p className="text-muted-foreground text-[10px] mt-0.5">Retail Intelligence</p>
            </div>
          )}
          <button onClick={() => setSidebarOpen((x) => !x)} className="ml-auto text-muted-foreground hover:text-foreground transition-colors p-1 rounded">
            {sidebarOpen ? <X size={15} /> : <Menu size={15} />}
          </button>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = active === item.id;
            return (
              <button key={item.id} onClick={() => setActive(item.id)} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${isActive ? "bg-primary/15 text-primary border border-primary/20" : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent"}`}>
                <Icon size={17} className="shrink-0" />
                {sidebarOpen && <span className="truncate">{item.label}</span>}
                {sidebarOpen && isActive && <ChevronRight size={13} className="ml-auto" />}
              </button>
            );
          })}
        </nav>

        <div className="p-3 border-t border-sidebar-border">
          {sidebarOpen ? (
            <div className="flex items-center gap-3 px-2 py-2">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center shrink-0">
                <User size={13} className="text-white" />
              </div>
              <div className="min-w-0">
                <p className="text-foreground text-xs font-semibold truncate">Nguyễn Minh Tuấn</p>
                <p className="text-muted-foreground text-[10px] truncate">Store Manager</p>
              </div>
              <button className="ml-auto text-muted-foreground hover:text-foreground p-1 rounded transition-colors">
                <LogOut size={13} />
              </button>
            </div>
          ) : (
            <div className="flex justify-center py-1">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center">
                <User size={13} className="text-white" />
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Header */}
        <header className="h-[60px] border-b border-border flex items-center px-6 gap-4 shrink-0 bg-background/80 backdrop-blur-sm">
          <div className="flex-1">
            <h1 className="text-foreground font-bold text-base leading-none">{sectionTitles[active].title}</h1>
            <p className="text-muted-foreground text-xs mt-0.5">{sectionTitles[active].sub}</p>
          </div>
          <div className="flex items-center gap-3">
            <input
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              placeholder="http://localhost:8000"
              className="hidden md:block text-xs px-2.5 py-1.5 rounded-lg bg-muted border border-border text-foreground/80 w-44 font-mono outline-none focus:border-primary/50"
              title="Địa chỉ backend API"
            />
            <span className="text-muted-foreground text-xs font-mono hidden sm:block">
              {new Date().toLocaleDateString("vi-VN", { weekday: "short", day: "2-digit", month: "2-digit", year: "numeric" })}
            </span>
            <button className="relative text-muted-foreground hover:text-foreground p-2 rounded-lg hover:bg-muted transition-colors">
              <Bell size={17} />
              <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-primary rounded-full" />
            </button>
            <button className="text-muted-foreground hover:text-foreground p-2 rounded-lg hover:bg-muted transition-colors">
              <Settings size={17} />
            </button>
          </div>
        </header>

        {/* Branch tab bar */}
        <div className="border-b border-border bg-background/60 backdrop-blur-sm px-6 overflow-x-auto scrollbar-hide">
          <div className="flex items-center gap-1 min-w-max py-2">
            <Building2 size={13} className="text-muted-foreground mr-1 shrink-0" />
            {branches.map((b) => {
              const isActive = branchId === b.id;
              return (
                <button key={b.id} onClick={() => setBranchId(b.id)} className={`relative flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${isActive ? "bg-primary/15 text-primary border border-primary/25" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}>
                  {b.id !== "all" && <MapPin size={10} className="shrink-0" />}
                  {b.label}
                  {isActive && b.id !== "all" && (<span className="text-[9px] font-normal text-primary/70 ml-0.5 hidden sm:inline">{b.city}</span>)}
                </button>
              );
            })}
          </div>
        </div>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-6">
          {active === "sales" && <SalesForecast branchId={branchId} apiBase={apiBase} productSales={topProducts} />}
          {active === "chatbot" && <AIChatbot aiContext={aiContext} />}
          {active === "analysis" && <ProductAnalysis pieData={pieData} />}
          {active === "products" && <ProductManagement />}
        </main>
      </div>
    </div>
  );
}