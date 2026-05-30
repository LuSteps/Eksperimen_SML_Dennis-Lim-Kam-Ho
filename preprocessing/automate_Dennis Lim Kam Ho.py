import pandas as pd


def preprocess_crypto_data(input_csv_path, output_csv_path):
    # Load dataset
    df = pd.read_csv(input_csv_path)

    # Konversi kolom tanggal
    df['date'] = pd.to_datetime(df['date'])

    # Urutkan berdasarkan tanggal
    df = df.sort_values('date')

    # Isi missing value market cap
    df['market_cap'] = df['market_cap'].ffill()

    # Hapus data duplikat
    df = df.drop_duplicates()

    # Feature extraction waktu
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter

    # Moving average
    for window in [7, 30]:
        df[f'price_ma_{window}d'] = df['price'].rolling(window=window).mean()
        df[f'total_volume_ma_{window}d'] = df['total_volume'].rolling(window=window).mean()
        df[f'market_cap_ma_{window}d'] = df['market_cap'].rolling(window=window).mean()

    # Feature lag harga
    for i in range(1, 4):
        df[f'price_lag_{i}'] = df['price'].shift(i)

    # Rolling mean harga
    df['price_rolling_mean_7'] = df['price'].rolling(window=7).mean()
    df['price_rolling_mean_30'] = df['price'].rolling(window=30).mean()

    # Hapus nilai kosong hasil rolling
    df = df.dropna()

    # Simpan hasil preprocessing
    df.to_csv(output_csv_path, index=False)

    print(f"Preprocessing selesai.")
    print(f"Output saved to: {output_csv_path}")


# Run
preprocess_crypto_data(
    input_csv_path="bitcoin_raw.csv",
    output_csv_path="preprocessing/bitcoin.csv"
)