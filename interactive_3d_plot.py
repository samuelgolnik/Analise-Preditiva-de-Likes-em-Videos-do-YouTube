import os
import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np
import statsmodels.api as sm # For fitting the log-transformed model to get plane coefficients
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import seaborn as sns # For styling (optional, but good practice)
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def get_db_connection():
    """Establishes and returns a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Successfully connected to the database for 3D plotting.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL for 3D plotting: {e}")
        print("Please check your database credentials and ensure PostgreSQL is running.")
        return None

def fetch_data_for_3d_plot(db_conn):
    """
    Fetches like_count, view_count, and comment_count from the database.
    """
    query = sql.SQL("""
        SELECT
            like_count,
            view_count,
            comment_count
        FROM
            video
        WHERE
            like_count IS NOT NULL AND view_count IS NOT NULL
            AND comment_count IS NOT NULL
            AND like_count >= 0 AND view_count >= 0 AND comment_count >= 0;
    """)
    
    try:
        df = pd.read_sql(query.as_string(db_conn), db_conn)
        print(f"Successfully fetched {len(df)} rows of data for 3D plot.")
        return df
    except Exception as e:
        print(f"Error fetching data for 3D plot: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def main():
    db_conn = None
    try:
        db_conn = get_db_connection()
        if not db_conn:
            return

        df = fetch_data_for_3d_plot(db_conn)

        if df.empty:
            print("No data available for 3D plot. Exiting.")
            return

        # --- Data Preparation for 3D Plot (Using Log Transformation for Visualization Clarity) ---
        # A 3D regression plane is best visualized with log-transformed data due to linearity assumptions.
        df_log_transformed = df.copy()
        skewed_vars = ['like_count', 'view_count', 'comment_count']
        for var in skewed_vars:
            if not (df_log_transformed[var] == 0).all():
                df_log_transformed[f'log_{var}'] = np.log1p(df_log_transformed[var])
            else:
                df_log_transformed[f'log_{var}'] = 0
        df_log_transformed.dropna(inplace=True)

        if df_log_transformed.empty:
            print("\nSkipping 3D plot: No valid data after log transformation for visualization.")
            return

        # Define dependent (Y) and independent (X) variables for the log-transformed model
        Y_log = df_log_transformed['log_like_count']
        X_log = df_log_transformed[['log_view_count', 'log_comment_count']]
        X_log = sm.add_constant(X_log) # Add constant for intercept

        # Fit a model on the log-transformed data specifically for the plane visualization
        model_log_transformed = sm.OLS(Y_log, X_log).fit()
        
        # Get coefficients from the log-transformed model for plotting the plane
        b0_log = model_log_transformed.params['const']
        b1_log = model_log_transformed.params['log_view_count']
        b2_log = model_log_transformed.params['log_comment_count']

        # --- Create the 3D Regression Plane Plot ---
        print("\n--- Generating 3D Regression Plane Plot (Log Transformed) ---")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of actual data points
        ax.scatter(df_log_transformed['log_view_count'],
                   df_log_transformed['log_comment_count'],
                   df_log_transformed['log_like_count'],
                   c='skyblue', marker='o', alpha=0.5, label='Vídeos')

        # Create a meshgrid for the regression plane
        x_surf, y_surf = np.meshgrid(np.linspace(df_log_transformed['log_view_count'].min(), df_log_transformed['log_view_count'].max(), 20),
                                     np.linspace(df_log_transformed['log_comment_count'].min(), df_log_transformed['log_comment_count'].max(), 20))
        
        # Calculate the predicted Z values (log_like_count) for the plane
        z_surf = b0_log + b1_log * x_surf + b2_log * y_surf

        # Plot the regression plane
        ax.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', alpha=0.6, label='Plano')

        # Set labels and title
        ax.set_xlabel('Número de Visualizações (Log)', fontsize=12)
        ax.set_ylabel('Número de Comentários (Log)', fontsize=12)
        ax.set_zlabel('Quantidade de Likes (Log)', fontsize=12)
        ax.set_title('Quantidade de Likes(Número de Visualizações, Número de Comentários)', fontsize=16)

        # Add a legend
        ax.legend()

        # You can adjust these values (elev=elevation, azim=azimuth) to change the view angle
        ax.view_init(elev=30, azim=120) 

        plt.tight_layout()
        
        # Show the plot interactively
        plt.show()
        
        # Close the plot to free memory after showing and saving
        plt.close(fig) 

    except Exception as e:
        print(f"An error occurred during 3D plotting: {e}")
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    main()
