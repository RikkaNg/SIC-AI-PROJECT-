import pandas as pd
import numpy as np

# 1. Định nghĩa các kiểu dữ liệu (dtypes) để tiết kiệm RAM khi đọc train.csv
dtypes_train = {
    'store_nbr': 'int8',
    'item_nbr': 'int32',
    'unit_sales': 'float32',
    'onpromotion': 'boolean'
}


# Đọc dữ liệu train (có thể mất vài phút do file rất lớn)
print("Đang đọc train.csv...")
train = pd.read_csv(r'E:\nic\SIC-AI-PROJECT-\data\raw\train.csv', parse_dates=['date'], dtype=dtypes_train)

# Lấy dữ liệu từ ngày 2016-01-01 trở đi để giảm dung lượng dữ liệu
train = train[train['date'] >= '2016-01-01'].copy()

# Đọc các file phụ trợ
print("Đang đọc các file phụ trợ...")
stores = pd.read_csv(r'E:\nic\SIC-AI-PROJECT-\data\raw\stores.csv')
items = pd.read_csv(r'E:\nic\SIC-AI-PROJECT-\data\raw\items.csv')
oil = pd.read_csv(r'E:\nic\SIC-AI-PROJECT-\data\raw\oil.csv', parse_dates=['date'])
holidays = pd.read_csv(r'E:\nic\SIC-AI-PROJECT-\data\raw\holidays_events.csv', parse_dates=['date'])
transactions = pd.read_csv(r'E:\nic\SIC-AI-PROJECT-\data\raw\transactions.csv', parse_dates=['date'])

# Tiền xử lý NaN của onpromotion
train['onpromotion'] = train['onpromotion'].fillna(False)
# Tiền xử lý NaN của transactions
# 2. Tiền xử lý dữ liệu Oil (Dầu mỏ)
# File oil.csv có thể thiếu ngày cuối tuần, tạo một dataframe ngày liên tục để merge
all_dates = pd.DataFrame(pd.date_range(train['date'].min(), train['date'].max()), columns=['date'])
oil = all_dates.merge(oil, on='date', how='left')
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill() # Điền giá trị thiếu bằng giá trị gần nhất
oil.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)

# 3. Tiền xử lý dữ liệu Holidays (Ngày lễ)
# Chỉ lấy các ngày lễ không bị chuyển (transferred = False) để tránh trùng lặp
holidays = holidays[holidays['transferred'] == False]
# Để đơn giản, ta chỉ lấy ngày lễ Quốc gia (National) và Regional để merge theo date
# (Bạn có thể mở rộng thêm logic theo 'locale_name' nếu muốn ghép theo từng thành phố/tỉnh)
nat_holidays = holidays[holidays['locale'] == 'National'][['date', 'type', 'description']]
nat_holidays = nat_holidays.rename(columns={
    'type': 'holiday_type', 
    'description': 'holiday_description'
})
nat_holidays = nat_holidays.drop_duplicates(subset=['date']) # Tránh trùng lặp ngày
# 4. Bắt đầu Merge dữ liệu vào Train
print("Đang ghép dữ liệu...")
# 4.1 Ghép thông tin cửa hàng (stores)
train = train.merge(stores, on='store_nbr', how='left')

# 4.2 Ghép thông tin sản phẩm (items)
train = train.merge(items, on='item_nbr', how='left')

# 4.3 Ghép giá dầu (oil)
train = train.merge(oil, on='date', how='left')

# 4.4 Ghép thông tin giao dịch (transactions) - Khóa ghép là cả date và store_nbr
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')

# 4.5 Ghép ngày lễ (holidays)
train = train.merge(nat_holidays, on='date', how='left')

# Điền giá trị NaN cho cột holiday_type (những ngày không phải lễ)
train['holiday_type'] = train['holiday_type'].fillna('Normal Day')
train['holiday_description'] = train['holiday_description'].fillna('None')

# 5. Kiểm tra kết quả
print("\nThông tin DataFrame sau khi ghép:")
print(train.info(memory_usage='deep'))
print("\n5 dòng đầu tiên:")
print(train.head())

# Lưu lại file (Tùy chọn, file sẽ rất nặng, khuyên dùng định dạng parquet để tiết kiệm dung lượng)
train.to_parquet(r'data\processed\train_merged.parquet', index=False)