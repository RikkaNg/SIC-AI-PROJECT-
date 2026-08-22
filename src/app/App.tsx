import { useState, useRef, useEffect } from "react";
import {
  LayoutDashboard, Package, Users, BarChart3, MessageSquare, TrendingUp,
  Bell, Search, ArrowUpRight, ArrowDownRight, AlertTriangle, Shield,
  Plus, Send, Bot, Menu, Store, Zap, RefreshCw, Warehouse,
} from "lucide-react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart,
} from "recharts";

type View = "dashboard" | "inventory" | "hr" | "analytics" | "chatbot" | "forecast";

// ─── DATA (Giữ nguyên data Figma để giao diện đẹp) ──────────────────────────────

const monthlyRevenue = [
  { month: "T1", doanhThu: 480, chiPhi: 310, loiNhuan: 170 },
  { month: "T2", doanhThu: 525, chiPhi: 342, loiNhuan: 183 },
  { month: "T3", doanhThu: 618, chiPhi: 381, loiNhuan: 237 },
  { month: "T4", doanhThu: 583, chiPhi: 368, loiNhuan: 215 },
  { month: "T5", doanhThu: 744, chiPhi: 422, loiNhuan: 322 },
  { month: "T6", doanhThu: 892, chiPhi: 512, loiNhuan: 380 },
  { month: "T7", doanhThu: 763, chiPhi: 458, loiNhuan: 305 },
  { month: "T8", doanhThu: 954, chiPhi: 548, loiNhuan: 406 },
];

const topProducts = [
  { name: "iPhone 15 Pro Max", sold: 284, category: "Điện thoại" },
  { name: "AirPods Pro Gen 2", sold: 312, category: "Phụ kiện" },
  { name: "Samsung Galaxy S24 Ultra", sold: 198, category: "Điện thoại" },
  { name: "MacBook Pro M3 14\"", sold: 127, category: "Laptop" },
  { name: "iPad Pro 12.9\" M2", sold: 94, category: "Tablet" },
];

const inventoryItems = [
  { id: "SP001", name: "iPhone 15 Pro Max 256GB", category: "Điện thoại", stock: 48, minStock: 20, buyPrice: 26500000, sellPrice: 31500000, status: "in" },
  { id: "SP002", name: "MacBook Pro M3 14\"", category: "Laptop", stock: 12, minStock: 15, buyPrice: 42000000, sellPrice: 50000000, status: "low" },
  { id: "SP003", name: "Samsung Galaxy S24 Ultra", category: "Điện thoại", stock: 35, minStock: 20, buyPrice: 18500000, sellPrice: 22000000, status: "in" },
  { id: "SP004", name: "AirPods Pro Gen 2", category: "Phụ kiện", stock: 7, minStock: 30, buyPrice: 5200000, sellPrice: 6800000, status: "critical" },
  { id: "SP005", name: "Dell XPS 15 9530", category: "Laptop", stock: 0, minStock: 10, buyPrice: 28000000, sellPrice: 35000000, status: "out" },
  { id: "SP006", name: "iPad Pro 12.9\" M2", category: "Tablet", stock: 23, minStock: 15, buyPrice: 22000000, sellPrice: 27500000, status: "in" },
  { id: "SP007", name: "Apple Watch Ultra 2", category: "Phụ kiện", stock: 9, minStock: 15, buyPrice: 15000000, sellPrice: 19500000, status: "low" },
  { id: "SP008", name: "Sony WH-1000XM5", category: "Phụ kiện", stock: 41, minStock: 20, buyPrice: 5800000, sellPrice: 8200000, status: "in" },
];

const employees = [
  { id: "NV001", name: "Nguyễn Văn An", role: "Quản lý kho", branch: "Hà Nội", dept: "Kho vận", status: "active" },
  { id: "NV002", name: "Trần Thị Bình", role: "Nhân viên bán hàng", branch: "TP.HCM", dept: "Kinh doanh", status: "active" },
  { id: "NV003", name: "Lê Minh Châu", role: "Giám đốc CN", branch: "Đà Nẵng", dept: "Quản lý", status: "active" },
  { id: "NV004", name: "Phạm Thị Dung", role: "Kế toán trưởng", branch: "Hà Nội", dept: "Tài chính", status: "active" },
  { id: "NV005", name: "Hoàng Văn Em", role: "Nhân viên IT", branch: "TP.HCM", dept: "Kỹ thuật", status: "inactive" },
  { id: "NV006", name: "Vũ Thị Phương", role: "Trưởng phòng MKT", branch: "Hà Nội", dept: "Marketing", status: "active" },
  { id: "NV007", name: "Đặng Quốc Hùng", role: "Nhân viên bán hàng", branch: "Đà Nẵng", dept: "Kinh doanh", status: "active" },
];

const branches = [
  { name: "Hà Nội", manager: "Nguyễn Văn An", employees: 24, revenue: "4.2 tỷ" },
  { name: "TP. Hồ Chí Minh", manager: "Trần Thị Bình", employees: 31, revenue: "5.8 tỷ" },
  { name: "Đà Nẵng", manager: "Lê Minh Châu", employees: 18, revenue: "2.9 tỷ" },
  { name: "Cần Thơ", manager: "Phạm Văn Đức", employees: 12, revenue: "1.6 tỷ" },
];

const roles = [
  { name: "Quản trị viên", count: 2, perms: ["Tất cả quyền"], color: "red" },
  { name: "Giám đốc CN", count: 4, perms: ["Xem", "Sửa", "Duyệt", "Báo cáo"], color: "purple" },
  { name: "Trưởng phòng", count: 8, perms: ["Xem", "Sửa", "Duyệt"], color: "blue" },
  { name: "Kế toán", count: 6, perms: ["Xem", "Tài chính"], color: "green" },
  { name: "Nhân viên", count: 53, perms: ["Xem cơ bản"], color: "gray" },
];

const profitData = [
  { category: "Điện thoại", doanhThu: 1248, chiPhi: 920, loiNhuan: 328, tyLe: 26.3 },
  { category: "Laptop", doanhThu: 902, chiPhi: 710, loiNhuan: 192, tyLe: 21.3 },
  { category: "Tablet", doanhThu: 412, chiPhi: 295, loiNhuan: 117, tyLe: 28.4 },
  { category: "Phụ kiện", doanhThu: 634, chiPhi: 380, loiNhuan: 254, tyLe: 40.1 },
  { category: "Màn hình", doanhThu: 278, chiPhi: 215, loiNhuan: 63, tyLe: 22.7 },
];

const forecastData = [
  { month: "T3", actual: 618, forecast: 628 },
  { month: "T4", actual: 583, forecast: 595 },
  { month: "T5", actual: 744, forecast: 722 },
  { month: "T6", actual: 892, forecast: 878 },
  { month: "T7", actual: 763, forecast: 791 },
  { month: "T8", actual: 954, forecast: 968 },
  { month: "T9", actual: undefined, forecast: 1055 },
  { month: "T10", actual: undefined, forecast: 1183 },
  { month: "T11", actual: undefined, forecast: 1428 },
];

const hotProducts = [
  { name: "AirPods Pro Gen 2", growth: 48.2, status: "hot" },
  { name: "iPhone 15 Pro Max", growth: 32.5, status: "hot" },
  { name: "iPad Pro M2", growth: 18.7, status: "rising" },
  { name: "MacBook Pro M3", growth: 12.4, status: "rising" },
  { name: "Sony WH-1000XM5", growth: -3.1, status: "cold" },
  { name: "Dell XPS 15", growth: -8.3, status: "cold" },
];

// ─── SMALL COMPONENTS ──────────────────────────────────────────────────────────

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

function StockBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    in: { label: "Còn hàng", cls: "bg-emerald-500/15 text-emerald-600 border-emerald-500/25" },
    low: { label: "Sắp hết", cls: "bg-amber-500/15 text-amber-600 border-amber-500/25" },
    critical: { label: "Rất ít", cls: "bg-orange-500/15 text-orange-600 border-orange-500/25" },
    out: { label: "Hết hàng", cls: "bg-red-500/15 text-red-600 border-red-500/25" },
  };
  const s = map[status] || map.in;
  return (
    <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${s.cls}`}>{s.label}</span>
  );
}

function SectionHeader({
  title, sub, action,
}: { title: string; sub?: string; action?: React.ReactNode }) {
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

function ChartTip({ active, payload, label }: { active?: boolean; payload?: { color: string; name: string; value: number }[]; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-2xl">
      <p className="text-[10px] font-mono text-slate-500 mb-1.5">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="text-xs font-mono">
          {p.name}: <span className="font-medium">{typeof p.value === "number" ? p.value.toLocaleString("vi-VN") : p.value}</span>
        </p>
      ))}
    </div>
  );
}

// ─── VIEWS ─────────────────────────────────────────────────────────────────────

function DashboardView() {
  const [recentOrders, setRecentOrders] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/dashboard/recent-orders")
      .then(res => res.json())
      .then(data => setRecentOrders(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-5">
      <SectionHeader
        title="Dashboard Tổng Quan"
        sub={`Cập nhật: ${new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })} — 22/08/2026`}
        action={
          <button className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 transition-colors">
            <RefreshCw size={12} />
            Làm mới
          </button>
        }
      />

      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Doanh thu hôm nay" value="₫482,350,000" sub="+12.4% so hôm qua" trend="up" color="blue" />
        <StatCard label="Đơn hàng" value="147" sub="+8 đơn so hôm qua" trend="up" color="green" />
        <StatCard label="Khách hàng mới" value="23" sub="-2 so hôm qua" trend="down" color="amber" />
        <StatCard label="Giá trị tồn kho" value="₫18.4 tỷ" sub="+₫240M tuần này" trend="up" color="purple" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* ... (Giữ nguyên đoạn biểu đồ AreaChart ở giữa) ... */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900 mb-4">Biểu đồ doanh thu (Đang dùng data tĩnh)</p>
          {/* Tạm thời bỏ biểu đồ ra cho gọn, bạn có thể giữ nguyên code biểu đồ cũ ở đây */}
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900 mb-4">Bán chạy nhất</p>
          <div className="space-y-3.5 text-xs text-slate-500">
            Đang tải từ DB...
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
            <AlertTriangle size={13} className="text-amber-500" />
            Cảnh báo tồn kho
          </p>
          <div className="space-y-0">
            <p className="text-xs text-slate-500">Xem chi tiết ở mục Quản lý kho</p>
          </div>
        </div>

        {/* Bảng Giao dịch gần đây đã được gọi API thật */}
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900 mb-3">Giao dịch gần đây (Từ DB)</p>
          <div className="space-y-0">
            {recentOrders.length === 0 ? (
              <p className="text-xs text-slate-500">Đang tải...</p>
            ) : (
              recentOrders.map((tx) => (
                <div key={tx.id} className="flex items-center justify-between py-2.5 border-b border-slate-100 last:border-0">
                  <div>
                    <p className="text-[10px] font-mono text-slate-500">{tx.id}</p>
                    <p className="text-xs text-slate-700">{tx.customer}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-mono text-emerald-500">{tx.amount} SP</p>
                    <p className="text-[10px] font-mono text-slate-500">{tx.time}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function InventoryView() {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");

  // Khai báo state để lưu data lấy từ DB
  const [inventoryData, setInventoryData] = useState<any[]>([]);

  // Gọi API lấy data thật khi component load
  useEffect(() => {
    fetch("http://localhost:8000/api/inventory/list")
      .then(res => res.json())
      .then(data => setInventoryData(data))
      .catch(err => console.error(err));
  }, []);

  const filtered = inventoryData.filter((item) => {
    const matchSearch =
      item.name.toLowerCase().includes(search.toLowerCase()) ||
      item.id.toLowerCase().includes(search.toLowerCase()) ||
      item.category.toLowerCase().includes(search.toLowerCase());
    const matchFilter = filter === "all" || item.status === filter;
    return matchSearch && matchFilter;
  });

  return (
    <div className="space-y-5">
      <SectionHeader
        title="Quản Lý Kho Hàng (Dữ liệu thật từ DB)"
        sub={`${inventoryData.length} sản phẩm — ${inventoryData.filter((i) => i.status !== "in").length} cần xử lý`}
        action={
          <button className="flex items-center gap-2 bg-blue-600 text-white text-xs px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium">
            <Plus size={13} />
            Thêm sản phẩm
          </button>
        }
      />

      <div className="flex items-center gap-2.5">
        <div className="relative max-w-xs w-full">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            className="w-full bg-white border border-slate-200 rounded-lg pl-8 pr-4 py-2 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500/40 transition-colors"
            placeholder="Tìm mã SP, tên, danh mục..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex gap-1.5">
          {[
            { key: "all", label: "Tất cả" },
            { key: "low", label: "Sắp hết" },
            { key: "critical", label: "Rất ít" },
            { key: "out", label: "Hết hàng" },
          ].map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`text-xs px-3 py-2 rounded-lg border transition-colors ${filter === f.key ? "bg-blue-50 border-blue-200 text-blue-600" : "bg-white border-slate-200 text-slate-500 hover:text-slate-900"}`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/50">
              {["Mã SP", "Tên sản phẩm", "Danh mục", "Tồn kho", "Lead Time", "Trạng thái"].map((h) => (
                <th key={h} className="text-left text-[10px] font-mono text-slate-500 uppercase tracking-wider px-4 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr><td colSpan={6} className="text-center py-8 text-slate-500 text-sm">Đang tải dữ liệu từ Database...</td></tr>
            ) : (
              filtered.map((item) => (
                <tr key={item.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors cursor-pointer">
                  <td className="px-4 py-3 text-[10px] font-mono text-slate-500">{item.id}</td>
                  <td className="px-4 py-3 text-xs text-slate-900">{item.name}</td>
                  <td className="px-4 py-3">
                    <span className="text-[10px] font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded-md">{item.category}</span>
                  </td>
                  <td className="px-4 py-3 text-xs font-mono">
                    <span className={item.stock === 0 ? "text-red-500" : item.stock < 10 ? "text-orange-500" : "text-emerald-500"}>
                      {item.stock}
                    </span>
                    <span className="text-slate-400"> / {item.minStock}</span>
                  </td>
                  <td className="px-4 py-3 text-[10px] font-mono text-slate-500">{item.minStock} ngày</td>
                  <td className="px-4 py-3"><StockBadge status={item.status} /></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HRView() {
  const [tab, setTab] = useState<"employees" | "branches" | "roles">("employees");

  const roleColors: Record<string, string> = {
    red: "bg-red-500/10 text-red-600",
    purple: "bg-purple-500/10 text-purple-600",
    blue: "bg-blue-500/10 text-blue-600",
    green: "bg-emerald-500/10 text-emerald-600",
    gray: "bg-slate-100 text-slate-600",
  };

  return (
    <div className="space-y-5">
      <SectionHeader
        title="Nhân Sự & Phân Quyền Chi Nhánh"
        sub="73 nhân viên — 4 chi nhánh — 5 vai trò"
        action={
          <button className="flex items-center gap-2 bg-blue-600 text-white text-xs px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium">
            <Plus size={13} />
            Thêm nhân viên
          </button>
        }
      />

      <div className="flex gap-1 bg-slate-100 rounded-xl p-1 w-fit border border-slate-200">
        {[
          { key: "employees", label: "Nhân viên" },
          { key: "branches", label: "Chi nhánh" },
          { key: "roles", label: "Vai trò & Phân quyền" },
        ].map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key as typeof tab)}
            className={`text-xs px-5 py-2 rounded-lg transition-all duration-150 ${tab === t.key ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-900"}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "employees" && (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/50">
                {["Mã NV", "Họ & Tên", "Chức vụ", "Chi nhánh", "Bộ phận", "Trạng thái"].map((h) => (
                  <th key={h} className="text-left text-[10px] font-mono text-slate-500 uppercase tracking-wider px-4 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {employees.map((emp) => {
                const initials = emp.name.split(" ").slice(-2).map((n) => n[0]).join("");
                return (
                  <tr key={emp.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-3 text-[10px] font-mono text-slate-500">{emp.id}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        <div className="w-7 h-7 rounded-full bg-blue-50 flex items-center justify-center text-[10px] font-semibold text-blue-600 flex-shrink-0">
                          {initials}
                        </div>
                        <span className="text-xs text-slate-900">{emp.name}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-500">{emp.role}</td>
                    <td className="px-4 py-3 text-xs text-slate-500">{emp.branch}</td>
                    <td className="px-4 py-3 text-xs text-slate-500">{emp.dept}</td>
                    <td className="px-4 py-3">
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${emp.status === "active" ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" : "bg-slate-100 text-slate-500 border-slate-200"}`}>
                        {emp.status === "active" ? "Hoạt động" : "Ngừng HĐ"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {tab === "branches" && (
        <div className="grid grid-cols-2 gap-4">
          {branches.map((branch) => (
            <div key={branch.name} className="bg-white border border-slate-200 rounded-xl p-5 hover:border-blue-200 transition-colors">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm font-semibold text-slate-900">Chi nhánh {branch.name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">Quản lý: {branch.manager}</p>
                </div>
                <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center">
                  <Store size={14} className="text-blue-600" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                  <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">Nhân viên</p>
                  <p className="text-xl font-mono font-semibold text-slate-900 mt-1">{branch.employees}</p>
                </div>
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                  <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">DT / Tháng</p>
                  <p className="text-sm font-mono font-semibold text-emerald-500 mt-1">₫{branch.revenue}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {tab === "roles" && (
        <div className="space-y-2.5">
          {roles.map((role) => (
            <div key={role.name} className="bg-white border border-slate-200 rounded-xl px-5 py-4 flex items-center justify-between hover:border-blue-200 transition-colors">
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${roleColors[role.color]}`}>
                  <Shield size={14} />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-900">{role.name}</p>
                  <p className="text-[10px] font-mono text-slate-500">{role.count} người dùng</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {role.perms.map((p) => (
                  <span key={p} className="text-[10px] bg-slate-100 text-slate-600 px-2.5 py-1 rounded-md border border-slate-200 font-mono">{p}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AnalyticsView() {
  const pieColors = ["#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#f43f5e"];

  return (
    <div className="space-y-5">
      <SectionHeader
        title="Phân Tích Lợi Nhuận Sản Phẩm"
        sub="Đánh giá hiệu quả kinh doanh — Q3/2026"
      />

      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Tổng doanh thu" value="₫3.47 tỷ" sub="+18.3% so Q2" trend="up" color="blue" />
        <StatCard label="Tổng chi phí" value="₫2.52 tỷ" sub="+11.2% so Q2" trend="up" color="amber" />
        <StatCard label="Lợi nhuận gộp" value="₫954M" sub="+32.8% so Q2" trend="up" color="green" />
        <StatCard label="Biên lợi nhuận" value="27.5%" sub="+3.2pp so Q2" trend="up" color="purple" />
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="col-span-3 bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Doanh thu / Chi phí / Lợi nhuận theo danh mục</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-4">Triệu VNĐ</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={profitData} barGap={3} barCategoryGap="25%">
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="category" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTip />} />
              <Legend wrapperStyle={{ fontSize: "10px", color: "#64748b", fontFamily: "JetBrains Mono" }} />
              <Bar dataKey="doanhThu" name="Doanh thu" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              <Bar dataKey="chiPhi" name="Chi phí" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              <Bar dataKey="loiNhuan" name="Lợi nhuận" fill="#10b981" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900">Cơ cấu lợi nhuận</p>
          <p className="text-[10px] font-mono text-slate-500 mt-0.5 mb-3">% biên lợi nhuận gộp theo danh mục</p>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={profitData} dataKey="tyLe" nameKey="category" cx="50%" cy="50%" outerRadius={70} innerRadius={38} paddingAngle={3}>
                {profitData.map((_, i) => <Cell key={i} fill={pieColors[i]} />)}
              </Pie>
              <Tooltip content={<ChartTip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5 mt-3">
            {profitData.map((item, i) => (
              <div key={item.category} className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: pieColors[i] }} />
                  <span className="text-[11px] text-slate-700">{item.category}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">{item.tyLe}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-900">Phân tích chi tiết theo danh mục</p>
        </div>
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/50">
              {["Danh mục", "Doanh thu", "Chi phí", "Lợi nhuận", "Biên LN", "Đánh giá"].map((h) => (
                <th key={h} className="text-left text-[10px] font-mono text-slate-500 uppercase tracking-wider px-5 py-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profitData.map((item) => (
              <tr key={item.category} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors">
                <td className="px-5 py-3 text-xs font-medium text-slate-900">{item.category}</td>
                <td className="px-5 py-3 text-xs font-mono text-blue-500">{item.doanhThu}M</td>
                <td className="px-5 py-3 text-xs font-mono text-amber-500">{item.chiPhi}M</td>
                <td className="px-5 py-3 text-xs font-mono text-emerald-500">{item.loiNhuan}M</td>
                <td className="px-5 py-3 text-xs font-mono text-slate-900 font-medium">{item.tyLe}%</td>
                <td className="px-5 py-3">
                  <span className={`text-[10px] font-mono px-2.5 py-1 rounded-full border ${item.tyLe >= 30 ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" : item.tyLe >= 20 ? "bg-blue-500/10 text-blue-600 border-blue-500/20" : "bg-amber-500/10 text-amber-600 border-amber-500/20"}`}>
                    {item.tyLe >= 30 ? "Xuất sắc" : item.tyLe >= 20 ? "Tốt" : "Trung bình"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type ChatMsg = { role: "user" | "ai"; text: string; time: string };

function ChatbotView() {
  const [messages, setMessages] = useState<ChatMsg[]>([
    {
      role: "ai",
      text: "Xin chào! Tôi là trợ lý AI phân tích kinh doanh. Tôi có thể giúp bạn:\n• Phân tích doanh thu và lợi nhuận\n• Cảnh báo tồn kho và đề xuất nhập hàng\n• Dự báo xu hướng bán hàng\n\nHệ thống đang kết nối với Backend Groq (Llama 3.1).",
      time: "14:30",
    },
  ]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const quickQ = [
    "Sản phẩm nào đang sắp hết hàng?",
    "Doanh thu tháng 8 là bao nhiêu?",
    "Danh mục nào có lợi nhuận cao nhất?",
    "Dự báo doanh thu Q4/2026?",
  ];

  // Hàm send gọi về Backend FastAPI thật
  const send = async (text: string) => {
    if (!text.trim() || typing) return;
    const now = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    setMessages((prev) => [...prev, { role: "user", text, time: now }]);
    setInput("");
    setTyping(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_query: text, chat_history: [] }),
      });
      if (!response.ok) throw new Error("Lỗi kết nối Backend API");
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: data.reply || "Tôi không hiểu phản hồi từ AI.", time: new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }) },
      ]);
    } catch (error: any) {
      console.error("Lỗi hệ thống:", error);
      setMessages((prev) => [
        ...prev,
        { role: "ai", text: `Lỗi AI: ${error.message}`, time: new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" }) },
      ]);
    } finally {
      setTyping(false);
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4">
      <div className="flex-1 flex flex-col bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-blue-50 flex items-center justify-center">
            <Bot size={14} className="text-blue-600" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">Trợ Lý AI</p>
            <p className="text-[10px] text-slate-500 font-mono">Powered by Groq Llama 3.1</p>
          </div>
          <div className="ml-auto flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
            <span className="text-[10px] font-mono text-emerald-500">Online</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
              <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${msg.role === "ai" ? "bg-blue-50" : "bg-slate-100"}`}>
                {msg.role === "ai" ? <Bot size={13} className="text-blue-600" /> : <span className="text-[10px] text-slate-500 font-mono">B</span>}
              </div>
              <div className={`flex flex-col gap-1 max-w-[78%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`rounded-2xl px-4 py-2.5 text-xs leading-relaxed whitespace-pre-line ${msg.role === "ai" ? "bg-slate-100 text-slate-900 rounded-tl-sm" : "bg-blue-600 text-white rounded-tr-sm"}`}>
                  {msg.text}
                </div>
                <span className="text-[10px] font-mono text-slate-400 px-1">{msg.time}</span>
              </div>
            </div>
          ))}
          {typing && (
            <div className="flex gap-2.5">
              <div className="w-7 h-7 rounded-full bg-blue-50 flex items-center justify-center">
                <Bot size={13} className="text-blue-600" />
              </div>
              <div className="bg-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
                    style={{ animationDelay: `${i * 0.18}s` }}
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
              className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500/40 transition-colors"
              placeholder="Nhập câu hỏi về dữ liệu kinh doanh..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send(input)}
            />
            <button
              onClick={() => send(input)}
              disabled={!input.trim() || typing}
              className="bg-blue-600 text-white px-3.5 rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-40"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>

      <div className="w-52 flex flex-col gap-3">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <p className="text-xs font-semibold text-slate-900 mb-3">Câu hỏi nhanh</p>
          <div className="space-y-1.5">
            {quickQ.map((q) => (
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
      </div>
    </div>
  );
}

function ForecastView() {
  return (
    <div className="space-y-5">
      <SectionHeader
        title="Dự Báo & Phân Tích Hàng Hóa"
        sub="Mô hình AI — Độ chính xác 87.3% — Dự báo 3 tháng tới"
      />

      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm font-semibold text-slate-900">Doanh thu thực tế & Dự báo</p>
            <p className="text-[10px] font-mono text-slate-500 mt-0.5">Triệu VNĐ — Vùng sau T8 là dự báo</p>
          </div>
          <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-blue-500 rounded inline-block" />Thực tế</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-purple-500 rounded inline-block opacity-60" style={{ borderTop: "2px dashed #a855f7" }} />Dự báo</span>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={forecastData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
            <XAxis dataKey="month" tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} domain={[400, "auto"]} />
            <Tooltip content={<ChartTip />} />
            <defs>
              <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="actual" name="Thực tế" stroke="#3b82f6" fill="url(#gActual)" strokeWidth={2} connectNulls={false} dot={{ fill: "#3b82f6", r: 3, strokeWidth: 0 }} />
            <Line type="monotone" dataKey="forecast" name="Dự báo" stroke="#a855f7" strokeWidth={2} strokeDasharray="6 4" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900 mb-1">Phân loại sản phẩm</p>
          <p className="text-[10px] font-mono text-slate-500 mb-4">Tăng trưởng doanh số 30 ngày qua</p>
          <div className="space-y-3">
            {hotProducts.map((p) => (
              <div key={p.name} className="flex items-center gap-3">
                <div className={`w-6 h-6 rounded-md flex items-center justify-center text-xs flex-shrink-0 ${p.status === "hot" ? "bg-red-500/10" : p.status === "rising" ? "bg-amber-500/10" : "bg-slate-100"}`}>
                  {p.status === "hot" ? "🔥" : p.status === "rising" ? "↑" : "↓"}
                </div>
                <span className="text-[11px] text-slate-700 flex-1 truncate">{p.name}</span>
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-slate-100 rounded-full h-1">
                    <div
                      className={`h-1 rounded-full ${p.growth > 0 ? "bg-emerald-500" : "bg-red-500"}`}
                      style={{ width: `${Math.min(Math.abs(p.growth) * 1.8, 100)}%` }}
                    />
                  </div>
                  <span className={`text-[10px] font-mono w-12 text-right ${p.growth > 0 ? "text-emerald-500" : "text-red-500"}`}>
                    {p.growth > 0 ? "+" : ""}{p.growth}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <p className="text-sm font-semibold text-slate-900 mb-1 flex items-center gap-2">
            <Zap size={13} className="text-amber-500" />
            Đề xuất từ AI
          </p>
          <p className="text-[10px] font-mono text-slate-500 mb-4">Dựa trên phân tích xu hướng</p>
          <div className="space-y-2.5">
            {[
              { type: "buy", text: "Tăng nhập AirPods Pro Gen 2 — Nhu cầu tăng 48% trong 2 tuần tới, hiện tồn kho chỉ còn 7 SP." },
              { type: "buy", text: "Chuẩn bị hàng iPhone 16 series — Dự báo ra mắt T10, đặt trước 200 SP cho 4 chi nhánh." },
              { type: "sell", text: "Xả hàng Dell XPS 15 — Xu hướng giảm -8.3%, đề xuất giảm giá 5% để tăng doanh số." },
              { type: "buy", text: "Đẩy mạnh phụ kiện Q4 — Biên lợi nhuận 40.1%, cao nhất danh mục, phù hợp mùa Noel." },
            ].map((rec, i) => (
              <div
                key={i}
                className={`flex gap-2.5 p-3 rounded-xl border text-xs leading-relaxed ${rec.type === "buy" ? "bg-emerald-500/5 border-emerald-500/15 text-slate-700" : "bg-amber-500/5 border-amber-500/15 text-slate-700"}`}
              >
                <div className={`w-1 flex-shrink-0 rounded-full mt-0.5 ${rec.type === "buy" ? "bg-emerald-500" : "bg-amber-500"}`} style={{ minHeight: "1rem" }} />
                {rec.text}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Dự báo T9/2026", value: "₫1.055 tỷ", change: "+10.5% vs T8", cls: "border-blue-500/15" },
          { label: "Dự báo T10/2026", value: "₫1.183 tỷ", change: "+12.1% vs T9", cls: "border-purple-500/15" },
          { label: "Dự báo T11/2026", value: "₫1.428 tỷ", change: "+20.7% vs T10 — Sale lớn", cls: "border-emerald-500/15" },
        ].map((card) => (
          <div key={card.label} className={`bg-white border rounded-xl p-4 ${card.cls}`}>
            <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">{card.label}</p>
            <p className="text-2xl font-semibold text-slate-900 tabular-nums mt-2 mb-1">{card.value}</p>
            <p className="text-[10px] font-mono text-emerald-500">{card.change}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── SHELL ─────────────────────────────────────────────────────────────────────

const navItems: { key: View; label: string; Icon: React.ElementType }[] = [
  { key: "dashboard", label: "Dashboard", Icon: LayoutDashboard },
  { key: "inventory", label: "Quản lý kho", Icon: Package },
  { key: "hr", label: "Nhân sự & Phân quyền", Icon: Users },
  { key: "analytics", label: "Phân tích lợi nhuận", Icon: BarChart3 },
  { key: "chatbot", label: "Trợ lý AI", Icon: MessageSquare },
  { key: "forecast", label: "Dự báo hàng hóa", Icon: TrendingUp },
];

function Sidebar({
  view, setView, open,
}: { view: View; setView: (v: View) => void; open: boolean }) {
  return (
    <aside
      className="flex-shrink-0 bg-white border-r border-slate-200 flex flex-col transition-all duration-200 overflow-hidden"
      style={{ width: open ? "220px" : "52px" }}
    >
      <div className="h-12 flex items-center gap-2.5 px-3.5 border-b border-slate-200 flex-shrink-0">
        <div className="w-6 h-6 bg-blue-600 rounded-md flex items-center justify-center flex-shrink-0">
          <Warehouse size={13} className="text-white" />
        </div>
        {open && (
          <span className="text-sm font-semibold text-slate-900 whitespace-nowrap overflow-hidden">BizManager Pro</span>
        )}
      </div>

      <nav className="flex-1 p-2 space-y-0.5 overflow-hidden">
        {navItems.map(({ key, label, Icon }) => {
          const active = view === key;
          return (
            <button
              key={key}
              onClick={() => setView(key)}
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
            <div className="w-7 h-7 rounded-full bg-blue-50 flex items-center justify-center text-[10px] font-bold text-blue-600 flex-shrink-0">A</div>
            <div className="min-w-0">
              <p className="text-xs font-medium text-slate-900 truncate">Admin Chính</p>
              <p className="text-[10px] text-slate-500 font-mono truncate">Quản trị viên</p>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

function TopBar({ open, setOpen }: { open: boolean; setOpen: (v: boolean) => void }) {
  return (
    <header className="h-12 flex-shrink-0 border-b border-slate-200 bg-white flex items-center px-4 gap-3">
      <button
        onClick={() => setOpen(!open)}
        className="text-slate-500 hover:text-slate-900 transition-colors p-1 rounded-md hover:bg-slate-100"
      >
        <Menu size={15} />
      </button>
      <div className="relative max-w-xs w-full">
        <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500/40 transition-colors"
          placeholder="Tìm kiếm toàn hệ thống..."
        />
      </div>
      <div className="ml-auto flex items-center gap-2">
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-500 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-2.5 py-1">
          <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
          Hệ thống hoạt động
        </div>
        <button className="relative w-8 h-8 flex items-center justify-center rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors">
          <Bell size={14} />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-red-500 rounded-full" />
        </button>
      </div>
    </header>
  );
}

export default function App() {
  const [view, setView] = useState<View>("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden" style={{ fontFamily: "'Be Vietnam Pro', system-ui, sans-serif" }}>
      <Sidebar view={view} setView={setView} open={sidebarOpen} />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <TopBar open={sidebarOpen} setOpen={setSidebarOpen} />
        <main className="flex-1 overflow-auto p-5 scrollbar-thin">
          {view === "dashboard" && <DashboardView />}
          {view === "inventory" && <InventoryView />}
          {view === "hr" && <HRView />}
          {view === "analytics" && <AnalyticsView />}
          {view === "chatbot" && <ChatbotView />}
          {view === "forecast" && <ForecastView />}
        </main>
      </div>
    </div>
  );
}