from tkinter import *
from tkinter.filedialog import *
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from preprocessing import normal
import numpy as np

def encode(df):
    
    encoder = OrdinalEncoder()
    encoded_data = encoder.fit_transform(df[['category']])
    encoded_df = df
    encoded_df['category'] = encoder.fit_transform(df[['category']])
    encoded_df['industry'] = encoder.fit_transform(df[['industry']])
    return encoded_df
    
# Read in the data and preprocess it as in the previous code
df = pd.read_csv('Book1updated.csv')
df = df.drop('Unnamed: 0', axis=1)
df = df.dropna(subset=['return'])
df = encode(df)
df = normal(df, 'return')
#encoder = OrdinalEncoder()
#encoded_data = encoder.fit_transform(df[['category']])
#encoded_df = df
#encoded_df['category'] = encoder.fit_transform(df[['category']])
#encoded_df['industry'] = encoder.fit_transform(df[['industry']])

#ridge regression
ridge_features = ['market_cap', 'net_profit_margin_inc', 'debt_to_equity',
                  'debt_to_equity', 'cash_flow_inc', 'inst_holding_change',
                  'price_to_earning', 'sign']
X_ridge = df[ridge_features]
y_ridge = df['return']
ridge = Ridge(alpha=1, solver='auto')
ridge.fit(X_ridge, y_ridge)

#lasso regression
from sklearn.linear_model import Lasso
lasso_features = ['sign','industry','ROCE','revenue_inc','cash_flow_inc']
X_lasso = df[lasso_features]
y_lasso = df['return']
lasso = Lasso(alpha=0.1)
lasso.fit(X_lasso, y_lasso)

#SVR
from sklearn.svm import SVR
svr_features = ['revenue_inc','sign','price_to_book','market_cap']
X_svr = df[svr_features]
y_svr = df['return']
svr = SVR(C= 1, gamma='scale', kernel ='rbf')
svr.fit(X_svr, y_svr)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf_features = ['revenue_inc','sign','price_to_book','market_cap']
X_rf = df[rf_features]
y_rf = df['return']
rf = RandomForestRegressor(max_depth = 5, max_features= 'sqrt', min_samples_leaf=1, min_samples_split= 2, n_estimators= 50,random_state=12)
rf.fit(X_rf, y_rf)

#elastic net
from sklearn.linear_model import ElasticNet
elastic_features = ['price_to_book', 'debt_to_equity','revenue_inc','cash_flow_inc', 'promoter_holding_change', 'inst_holding_change','sign']
X_elastic = df[elastic_features]
y_elastic = df['return']
elastic = ElasticNet(alpha = 10, l1_ratio = 1)
elastic.fit(X_elastic, y_elastic)
def output(book_path):
    test = pd.read_csv(book_path)
    test = test.drop('Unnamed: 0', axis=1)
    test = encode(test)
    if selected_algorithm == "Ridge Regression":
        test_x = test[ridge_features]
        test_y_pred = ridge.predict(test_x)
    elif selected_algorithm == "Lasso Regression":
        test_x = test[lasso_features]
        test_y_pred = lasso.predict(test_x)
    elif selected_algorithm == "SVR":
        test_x = test[svr_features]
        test_y_pred = svr.predict(test_x)
    elif selected_algorithm == "Random Forest":
        test_x = test[rf_features]
        test_y_pred = rf.predict(test_x)
    elif selected_algorithm == "Elastic Net":
        test_x = test[elastic_features]
        test_y_pred = elastic.predict(test_x)
    return test_y_pred


# Define a function to open a file dialog and return the selected file path
def browse_file():
    file_path = filedialog.askopenfilename()
    return file_path

def get_actual_returns(book_path):
    df = pd.read_csv(book_path)
    company_names = df['company']
    actual_returns = df['return']
    return company_names, actual_returns



# Define a function to handle the button click event
def calculate_return():
    book_path = browse_file()
    company_names, actual_returns = get_actual_returns(book_path)

    # Use the selected algorithm to predict the returns
    predicted_returns = output(book_path)

    # Compute the absolute mean of the difference between the actual and predicted returns
    abs_mean_diff = abs(actual_returns - predicted_returns).mean()

    # Create a new window to display the returns and the absolute mean difference
    results_window = tk.Toplevel(window)
    results_window.title('Predicted and Actual Returns')
    results_window.geometry('800x800')
    algorithm_label = tk.Label(results_window, text=f'Selected algorithm: {selected_algorithm}')
    algorithm_label.pack()

    returns_frame = tk.LabelFrame(results_window, text='Returns', padx=10, pady=10)
    returns_frame.pack(pady=10)

    # Create Labels inside the LabelFrame to display the predicted and actual returns for each company
    for i, company in enumerate(company_names):
        # Create a Label for the company name
        name_label = tk.Label(returns_frame, text=company)
        name_label.grid(row=i, column=0, padx=5, pady=5)

        # Create a Label for the predicted return
        predicted_label = tk.Label(returns_frame, text=f'Predicted: {predicted_returns[i]:.2f}')
        predicted_label.grid(row=i, column=1, padx=5, pady=5)

        # Create a Label for the actual return
        actual_label = tk.Label(returns_frame, text=f'Actual: {actual_returns[i]:.2f}')
        actual_label.grid(row=i, column=2, padx=5, pady=5)

    # Create the figure and axis for the bar graph
    fig, ax = plt.subplots()
    x = np.arange(len(company_names))
    bar_width = 0.35
    actual_bar = ax.bar(x - (bar_width/2), actual_returns, width=bar_width, label='Actual')
    predicted_bar = ax.bar(x + bar_width/2, predicted_returns, width=bar_width, label='Predicted')
    
    # Set the x coordinates for the bars
    
    
    # Set the width of the bars
    
    
    # Create the bars for predicted and actual returns
    #predicted_bar = ax.bar(x, predicted_returns, width, label='Predicted Returns')
    #actual_bar = ax.bar(x, actual_returns, width, label='Actual Returns')
    
    # Add labels, title, and legend to the graph
    ax.set_ylabel('Returns')
    ax.set_title('Comparison of Predicted and Actual Values')
    ax.set_xticks(x)
    plt.axhline(0, color='black', linewidth=0.5)
    ax.set_xticklabels(company_names)
    ax.legend()

    # Add the graph to the new window
    canvas = FigureCanvasTkAgg(fig, master=results_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Add a label to display the absolute mean difference
    abs_mean_label = tk.Label(results_window, text=f'Absolute mean difference: {abs_mean_diff:.2f}')
    abs_mean_label.pack()

    

def select_algorithm(algorithm):
    global selected_algorithm
    selected_algorithm = algorithm


# Create a menu with options for selecting the algorithm





# Create the main window and widgets
window = tk.Tk()
window.title('Stock Return Estimator')
window.geometry('400x400')

menu = tk.Menu(window)


algorithm_menu = tk.Menu(menu, tearoff=0)
algorithm_menu.add_command(label="Random Forest", command=lambda: select_algorithm("Random Forest"))
algorithm_menu.add_command(label="SVR", command=lambda: select_algorithm("SVR"))
algorithm_menu.add_command(label="Ridge Regression", command=lambda: select_algorithm("Ridge Regression"))
algorithm_menu.add_command(label="Lasso Regression", command=lambda: select_algorithm("Lasso Regression"))
algorithm_menu.add_command(label="Elastic Net", command=lambda: select_algorithm("Elastic Net"))
menu.add_cascade(label="Select Algorithm", menu=algorithm_menu)
window.config(menu=menu)

label = tk.Label(window, text='Select a CSV file:')
label.pack(pady=10)

button = tk.Button(window, text='Browse', command=calculate_return)
button.pack(pady=10)

result_label = tk.Label(window, text='')
result_label.pack(pady=10)

window.mainloop()