import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Biến toàn cục để lưu trữ dữ liệu CSV đã đọc
df_global = None
label_encoders = {}  # Lưu trữ các bộ mã hóa nhãn cho các cột


def read_csv():
    global df_global
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df_global = pd.read_csv(file_path)

        for i in tree.get_children():
            tree.delete(i)
        tree["column"] = list(df_global.columns)
        tree["show"] = "headings"
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        for index, row in df_global.head(30).iterrows():
            tree.insert("", "end", values=list(row))
        btn_histogram['state'] = 'normal'
        btn_boxplot['state'] = 'normal'
        btn_clear['state'] = 'normal'
        btn_forecast['state'] = 'normal'


def describe_data():
    global df_global
    if df_global is not None:
        description = df_global.describe().reset_index()
        for i in tree.get_children():
            tree.delete(i)
        tree["column"] = list(description.columns)
        tree["show"] = "headings"
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        for index, row in description.iterrows():
            tree.insert("", "end", values=list(row))
    else:
        messagebox.showwarning("Cảnh báo", "Vui lòng đọc file CSV trước khi phân tích mô tả.")


def plot_histogram():
    global df_global
    if df_global is not None:
        columns = df_global.columns.tolist()

        def plot_single_histogram(column):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df_global[column], bins=5, edgecolor='k')
            ax.set_title(f"Histogram của {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

            hist_window = tk.Toplevel(root)
            hist_window.title(f"Histogram của {column}")
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig)

        for column in columns:
            plot_single_histogram(column)


def plot_boxplot():
    global df_global
    if df_global is not None:
        data = df_global.select_dtypes(include=['number'])
        columns = data.columns.tolist()

        def plot_single_boxplot(column):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(data[column])
            ax.set_title(f'Box Plot của {column}')
            ax.set_ylabel(column)
            ax.grid(True)

            boxplot_window = tk.Toplevel(root)
            boxplot_window.title(f'Box Plot của {column}')
            canvas = FigureCanvasTkAgg(fig, master=boxplot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig)

        for column in columns:
            plot_single_boxplot(column)


def clear_data():
    global df_global, label_encoders
    df_global = None
    label_encoders = {}
    for i in tree.get_children():
        tree.delete(i)
    tree["column"] = []
    tree["show"] = ""
    btn_histogram['state'] = 'disabled'
    btn_boxplot['state'] = 'disabled'
    btn_clear['state'] = 'disabled'
    btn_forecast['state'] = 'disabled'


# def forecast_data():
#
#     global df_global, label_encoders
#     if df_global is not None:
#         input_data = []
#         input_window = tk.Toplevel(root)
#         input_window.title("Nhập dữ liệu dự báo")
#
#         labels = []
#         entries = []
#         for column in df_global.columns[:-1]:
#             label = tk.Label(input_window, text=column)
#             label.pack()
#             labels.append(label)
#             entry = tk.Entry(input_window)
#             entry.pack()
#             entries.append(entry)
#
#         def get_input_data():
#             nonlocal input_data
#             input_data = [entry.get() for entry in entries]
#             input_window.destroy()
#
#         submit_button = tk.Button(input_window, text="Dự Báo", command=get_input_data)
#         submit_button.pack()
#
#         input_window.wait_window(input_window)
#
#         input_data = [float(value) if value.isdigit() else value for value in input_data]
#         for i, column in enumerate(df_global.columns[:-1]):
#             if column in df_global.select_dtypes(include=['object']).columns:
#                 if column not in label_encoders:
#                     le = LabelEncoder()
#                     df_global[column] = le.fit_transform(df_global[column])
#                     label_encoders[column] = le
#                 input_data[i] = label_encoders[column].transform([input_data[i]])[0]
#
#         X_new = np.array(input_data).reshape(1, -1)
#         X = df_global.iloc[:, :-1].values
#         y = df_global.iloc[:, -1].values
#
#         model = LinearRegression()
#         model.fit(X, y)
#
#         y_pred = model.predict(X_new)
#
#         messagebox.showinfo("Dự Báo", f"Giá trị dự báo: {y_pred[0]}")
#     else:
#         messagebox.showwarning("Cảnh báo", "Vui lòng đọc file CSV trước khi dự báo.")
def forecast_data():
    global df_global, label_encoders
    if df_global is not None:
        input_data = []
        input_window = tk.Toplevel(root)
        input_window.title("Nhập dữ liệu dự báo")

        labels = []
        entries = []
        for column in df_global.columns[:-1]:
            label = tk.Label(input_window, text=column)
            label.pack()
            labels.append(label)
            entry = tk.Entry(input_window, width=50)
            entry.pack()
            entries.append(entry)

        def get_input_data():
            nonlocal input_data
            input_data = [entry.get() for entry in entries]
            input_window.destroy()

        submit_button = tk.Button(input_window, text="Predict", command=get_input_data)
        submit_button.pack()

        input_window.wait_window(input_window)

        # Xử lý dữ liệu đầu vào
        processed_data = []
        for i, column in enumerate(df_global.columns[:-1]):
            value = input_data[i]
            if df_global[column].dtype == 'object':  # Dữ liệu phân loại
                if column in label_encoders:
                    # Kiểm tra giá trị mới
                    try:
                        processed_value = label_encoders[column].transform([value])[0]
                    except ValueError:
                        # Nếu giá trị mới không tồn tại, ánh xạ thành giá trị đặc biệt
                        processed_value = -1
                else:
                    # Tạo bộ mã hóa mới nếu chưa tồn tại
                    le = LabelEncoder()
                    df_global[column] = le.fit_transform(df_global[column])
                    label_encoders[column] = le
                    try:
                        processed_value = le.transform([value])[0]
                    except ValueError:
                        processed_value = -1
                processed_data.append(processed_value)
            else:
                # Xử lý dữ liệu số
                try:
                    processed_value = float(value)
                except ValueError:
                    processed_value = 0  # Giá trị mặc định nếu không hợp lệ
                processed_data.append(processed_value)

        # Chuẩn hóa dữ liệu (nếu cần)
        scaler = MinMaxScaler()  # Hoặc StandardScaler
        X = df_global.iloc[:, :-1].values
        X_scaled = scaler.fit_transform(X)  # Áp dụng chuẩn hóa cho dữ liệu train
        X_new = scaler.transform([processed_data])  # Chuẩn hóa dữ liệu đầu vào

        # Mô hình dự đoán
        y = df_global.iloc[:, -1].values
        model = LinearRegression()
        model.fit(X_scaled, y)

        y_pred = model.predict(X_new)

        messagebox.showinfo("Dự Báo", f"Giá trị dự báo: {y_pred[0]}")
    else:
        messagebox.showwarning("Cảnh báo", "Vui lòng đọc file CSV trước khi dự báo.")


def quit_app():
    root.quit()


# Tạo cửa sổ chính
root = tk.Tk()
root.title("Ứng dụng phân tích dữ liệu")
root.geometry("800x600")

# Tạo tiêu đề
title = tk.Label(root, text="Phân tích dữ liệu CSV", font=("Helvetica", 16), pady=20)
title.pack()

# Tạo vùng hiển thị Treeview với kiểu scrollbar
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

tree = ttk.Treeview(frame)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tree.configure(yscrollcommand=scrollbar.set)

# Tạo khung chứa các nút để dễ dàng căn chỉnh
btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

# Tạo các nút
btn_read = tk.Button(btn_frame, text="Đọc file CSV", command=read_csv, bg="#4CAF50", fg="white", padx=10, pady=5)
btn_read.grid(row=0, column=0, padx=10)

btn_describe = tk.Button(btn_frame, text="Phân tích mô tả", command=describe_data, bg="#008CBA", fg="white", padx=10,
                         pady=5)
btn_describe.grid(row=0, column=1, padx=10)

btn_histogram = tk.Button(btn_frame, text="Vẽ Histogram", command=plot_histogram, bg="#FFA500", fg="white", padx=10,
                          pady=5, state='disabled')
btn_histogram.grid(row=0, column=2, padx=10)

btn_boxplot = tk.Button(btn_frame, text="Vẽ Boxplot", command=plot_boxplot, bg="#FF6347", fg="white", padx=10, pady=5,
                        state='disabled')
btn_boxplot.grid(row=0, column=3, padx=10)

btn_clear = tk.Button(btn_frame, text="Clear", command=clear_data, bg="#D3D3D3", fg="black", padx=10, pady=5,
                      state='disabled')
btn_clear.grid(row=0, column=4, padx=10)

btn_forecast = tk.Button(btn_frame, text="Dự Báo", command=forecast_data, bg="#FFA500", fg="white", padx=10, pady=5,
                         state='disabled')
btn_forecast.grid(row=0, column=5, padx=10)

btn_quit = tk.Button(btn_frame, text="Thoát", command=quit_app, bg="#f44336", fg="white", padx=10, pady=5)
btn_quit.grid(row=0, column=6, padx=10)

# Chạy ứng dụng Tkinter
root.mainloop()
