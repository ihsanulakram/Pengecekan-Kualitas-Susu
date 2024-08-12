import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from datetime import datetime
import os

# @st.cache_data # fungsi 'load_data' akan dipanggil sekali saja saat pertama kali aplikasi dijalankan
def load_data():
    data = pd.read_csv('data/milkdata.csv')
    return data

def prepare_data(data):
    x = data.drop('Grade', axis=1).values
    y = data['Grade'].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, le, scaler

def train_model(x_train, y_train):
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    return nb

def list_csv_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return files

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def plot_bar_chart(stats_df, x_col, y_col, title, colors, format_text):
    fig = px.bar(stats_df, x=x_col, y=y_col, title=title, text=y_col, color=x_col, color_discrete_sequence=colors)
    fig.update_traces(texttemplate=format_text, textposition='outside', cliponaxis=False)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, xaxis_tickangle=0)
    st.plotly_chart(fig)

def plot_pie_chart(labels, values, title, colors):
    fig = px.pie(values=values, names=labels, title=title, color_discrete_sequence=colors)
    fig.update_traces(textinfo='percent+label', textposition='inside')
    st.plotly_chart(fig)

# Load data
data = load_data()

# Menyiapkan Data
x, y, le, scaler = prepare_data(data)

# Membagi Dataset menjadi Data Training dan Data Testing
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Melatih Model Naive Bayes
# nb = train_model(x_train, y_train) # Penggunaan apabila data training dan data testing di spit
nb = train_model(x, y)

# Prediksi
# y_pred = nb.predict(x_test)

# Menghitung Akurasi
# accuracy = np.sum(y_test == y_pred) / len(y_test)

# Menu
with st.sidebar:
    selected = option_menu(
        'Menu', ['Pengecekan Grade Susu', 'Data Susu', 'Informasi Program'],
        icons=['check-circle', 'database', 'info-circle'], menu_icon='list', default_index=0
    )

if selected == 'Pengecekan Grade Susu':
    # Penggunaan Model pada Data Baru
    st.title('Pengecekan Grade Susu')    
    st.markdown('Silakan masukkan nilai-nilai berikut untuk mengecek grade susu:')
    # Form Input
    with st.form(key='input_form'):
        nilai_pH = st.number_input('**pH (0-14)**', format='%.1f', min_value=0.0, max_value=14.0, step=0.1)
        nilai_Temprature = st.number_input('**Temprature (\u00B0C)**', min_value=0, max_value=99, step=1)
        nilai_Taste = st.number_input('**Taste (0 = Bad, 1 = Good)**', min_value=0, max_value=1)
        nilai_Odor = st.number_input('**Odor (0 = Bad, 1 = Good)**', min_value=0, max_value=1)
        nilai_Fat = st.number_input('**Fat (0 = Bad, 1 = Good)**', min_value=0, max_value=1)
        nilai_Turbidity = st.number_input('**Turbidity (0 = Bad, 1 = Good)**', min_value=0, max_value=1)
        nilai_Colour = st.slider('**Colour**', 240, 255, key='nilai_Colour')
        submit_button = st.form_submit_button(label='CEK')

    if submit_button:
        data_baru = np.array([[nilai_pH, nilai_Temprature, nilai_Taste, nilai_Odor, nilai_Fat, nilai_Turbidity, nilai_Colour]])
        data_baru_scaled = scaler.transform(data_baru)
        prediksi = nb.predict(data_baru_scaled)
        grade_susu = le.inverse_transform(prediksi)[0]

        st.write('## Grade Susu:')
        if grade_susu == 'high':
            st.success(grade_susu.upper())
        elif grade_susu == 'medium':
            st.warning(grade_susu.upper())
        else:
            st.error(grade_susu.upper())

        # Simpan data ke file CSV
        result_df = pd.DataFrame(data_baru, columns=['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour'])
        result_df['Grade'] = grade_susu
        
        # Mengatur tipe data kolom
        result_df = result_df.astype({
            'pH': 'float',
            'Temprature': 'int',
            'Taste': 'int',
            'Odor': 'int',
            'Fat': 'int',
            'Turbidity': 'int',
            'Colour': 'int'
        })
        
        today = datetime.today().strftime('%Y-%m-%d')
        file_name = f'data/data_susu_{today}.csv'

        # Update Data Latih
        update_data = f'data/milkdata.csv'

        if os.path.exists(file_name):
            result_df.to_csv(file_name, mode='a', header=False, index=False)
            result_df.to_csv(update_data, mode='a', header=False, index=False)
        else:
            result_df.to_csv(file_name, mode='a', header=True, index=False)
            result_df.to_csv(update_data, mode='a', header=False, index=False)

        st.write('## Data Telah Disimpan')
        st.write(result_df)

elif selected == 'Data Susu':
    st.title('Data Susu')

    # Menampilkan file CSV yang ada pada directory
    directory = 'data'
    csv_files = list_csv_files(directory)

    selected_file = st.selectbox('Pilih file CSV', csv_files)
    if selected_file:
        file_path = os.path.join(directory, selected_file)
        data = load_csv_data(file_path)
        st.dataframe(data)

        # Download Data CSV
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="Unduh Data",
            data=csv_data,
            file_name=selected_file,
            mime='text/csv',
        )

        st.write(f'Jumlah Data : {len(data)}')
        
        # Statistik pH
        pH_stats = {
            'Statistik': ['Average', 'Max', 'Min'],
            'pH': [data['pH'].mean(), data['pH'].max(), data['pH'].min()]
        }
        pH_stats_df = pd.DataFrame(pH_stats)
        pH_colors = ['#636EFA', '#FF7F0E', '#2CA02C']
        plot_bar_chart(pH_stats_df, 'Statistik', 'pH', 'Statistik pH', pH_colors, '%{text:.1f}')

        # Statistik Temperature
        temp_stats = {
            'Statistik': ['Average', 'Max', 'Min'],
            'Suhu (\u00B0C)': [data['Temprature'].mean(), data['Temprature'].max(), data['Temprature'].min()]
        }
        temp_stats_df = pd.DataFrame(temp_stats)
        temp_colors = ['#636EFA', '#FF7F0E', '#2CA02C']
        plot_bar_chart(temp_stats_df, 'Statistik', 'Suhu (\u00B0C)', 'Statistik Temprature', temp_colors, '%{text:.0f}')

        # Statistik Taste, Odor, Fat, dan Turbidity
        stats = {
            'Statistik': ['Taste = 0', 'Taste = 1', 'Odor = 0', 'Odor = 1', 'Fat = 0', 'Fat = 1', 'Turbidity = 0', 'Turbidity = 1'],
            'Nilai': [
                data['Taste'].value_counts().get(0, 0), data['Taste'].value_counts().get(1, 0),
                data['Odor'].value_counts().get(0, 0), data['Odor'].value_counts().get(1, 0),
                data['Fat'].value_counts().get(0, 0), data['Fat'].value_counts().get(1, 0),
                data['Turbidity'].value_counts().get(0, 0), data['Turbidity'].value_counts().get(1, 0)
            ]
        }
        stats_df = pd.DataFrame(stats)
        stats_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF69B4', '#7FFF00', '#D2691E']
        plot_bar_chart(stats_df, 'Statistik', 'Nilai', 'Statistik Taste, Odor, Fat, dan Turbidity (0 = Bad, 1 = Good)', stats_colors, '%{text:.0f}')

        # Statistik Colour
        colour_stats = {
            'Statistik': ['Average', 'Max', 'Min'],
            'Nilai': [data['Colour'].mean(), data['Colour'].max(), data['Colour'].min()]
        }
        colour_stats_df = pd.DataFrame(colour_stats)
        colour_colors = ['#636EFA', '#FF7F0E', '#2CA02C']
        plot_bar_chart(colour_stats_df, 'Statistik', 'Nilai', 'Statistik Colour', colour_colors, '%{text:.0f}')

        # Statistik Grade
        grade_stats = data['Grade'].value_counts()
        grade_colors = ['#636EFA', '#FF7F0E', '#2CA02C']
        plot_pie_chart(grade_stats.index, grade_stats.values, 'Distribusi Grade', grade_colors)
    else:
        st.write('Tidak ada file CSV yang ditemukan.')

elif selected == 'Informasi Program':
    st.title('Informasi Program')
    st.write('\u25CF Dataset yang digunakan :')
    st.dataframe(data)
    st.text(f'\u25CF Jumlah Data          : {len(data)}')
    # st.text(f'\u25CF Pembagian Dataset    :')
    # st.text(f'- Data Training : {len(x_train)} (80%)')
    # st.text(f'- Data Testing  : {len(x_test)} (20%)')
    # st.text(f'\u25CF Akurasi Program      : {accuracy * 100:.2f}%')
    # st.text(f'\u25CF Laporan Klasifikasi  :\n{classification_report(y_test, y_pred, target_names=le.classes_)}')
    st.text(f'\u25CF Laporan Klasifikasi  :\n{classification_report(y, y, target_names=le.classes_)}')
