"""Modul utilitas untuk memisahkan nama lengkap menjadi nama depan dan belakang.

Praktik terbaik diterapkan:
- Penamaan variabel mengikuti standar PEP 8 (snake_case).
- Fungsi modular yang dapat digunakan kembali (reusable) dan dilengkapi type hints.
- Penanganan jika nama memiliki lebih dari dua kata.
- Guard block `if __name__ == '__main__':` untuk eksekusi skrip mandiri.
"""

from typing import Optional
import pandas as pd


def split_full_name(
    df: pd.DataFrame,
    name_col: str = "Name",
    first_name_col: str = "First",
    last_name_col: str = "Last",
) -> pd.DataFrame:
    """Memisahkan kolom nama lengkap menjadi nama depan dan nama belakang.

    Args:
        df (pd.DataFrame): DataFrame yang memiliki kolom nama.
        name_col (str): Nama kolom yang berisi nama lengkap. Default 'Name'.
        first_name_col (str): Nama kolom untuk nama depan. Default 'First'.
        last_name_col (str): Nama kolom untuk nama belakang. Default 'Last'.

    Returns:
        pd.DataFrame: DataFrame baru dengan kolom nama depan dan belakang.
    """
    if name_col not in df.columns:
        raise KeyError(f"Kolom '{name_col}' tidak ditemukan dalam DataFrame.")

    # Membuat salinan dataframe agar tidak memodifikasi dataframe asal (immutability)
    result_df = df.copy()

    # Memisahkan nama depan dan nama belakang (maksimal 1 kali split agar nama belakang menampung sisanya)
    split_names = result_df[name_col].astype(str).str.strip().str.split(" ", n=1, expand=True)

    result_df[first_name_col] = split_names[0]
    # Jika nama hanya 1 kata, isi string kosong atau None untuk nama belakang
    result_df[last_name_col] = split_names[1].fillna("")

    return result_df


def main() -> None:
    """Fungsi utama untuk mendemonstrasikan pemisahan kolom nama."""
    # Data dummy pengembang Machine Learning di Indonesia
    developer_data = {
        "Name": ["Isyana Saraswati", "Nicholas Saputra", "Raisa Andriana"],
        "Age": [30, 36, 32],
    }

    developer_df = pd.DataFrame(developer_data)
    print("=== Data Developer Asli ===")
    print(developer_df.to_string(index=False))

    # Pemisahan nama menggunakan fungsi
    processed_df = split_full_name(developer_df, name_col="Name")

    print("\n=== Data Setelah Kolom 'Name' Dipisahkan ===")
    print(processed_df.to_string(index=False))


if __name__ == "__main__":
    main()