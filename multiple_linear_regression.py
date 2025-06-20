import os
import pandas as pd
import psycopg2
from psycopg2 import sql
import numpy as np
import statsmodels.api as sm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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
        print("Successfully connected to the database for regression analysis.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL for regression analysis: {e}")
        print("Please check your database credentials and ensure PostgreSQL is running.")
        return None

def fetch_video_data_for_regression(db_conn):
    """
    Fetches necessary video data from the database for multiple linear regression.
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
        print(f"Successfully fetched {len(df)} rows of data for regression analysis.")
        return df
    except Exception as e:
        print(f"Error fetching data for regression analysis: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def main():
    db_conn = None
    try:
        db_conn = get_db_connection()
        if not db_conn:
            return

        df = fetch_video_data_for_regression(db_conn)

        if df.empty:
            print("No data available for regression analysis. Exiting.")
            return

        # --- Data Preparation for Regression Model ---
        regression_df_untransformed = df[['like_count', 'view_count', 'comment_count']].copy()
        regression_df_untransformed.dropna(inplace=True)

        if regression_df_untransformed.empty:
            print("No valid data for untransformed regression after cleaning. Exiting.")
            return

        Y_untransformed = regression_df_untransformed['like_count']
        X_untransformed = regression_df_untransformed[['view_count', 'comment_count']]
        X_untransformed = sm.add_constant(X_untransformed)

        # --- Build and Fit the Multiple Linear Regression Model ---
        print("\n--- Building and Fitting Multiple Linear Regression Model (No Log Transformation) ---")
        model_untransformed = sm.OLS(Y_untransformed, X_untransformed).fit()
        print(model_untransformed.summary())

        print("\n--- Interpretation Guidelines for Regression Summary ---")
        print("R-squared: Proportion of variance in the dependent variable (Likes) predictable from the independent variables.")
        print("Adj. R-squared: R-squared adjusted for the number of predictors. Better for comparing models.")
        print("Coefficients (coef): The change in the dependent variable for a one-unit change in the independent variable, holding others constant.")
        print("P>|t| (p-value): The probability that the coefficient is zero (no effect). A low p-value (<0.05) suggests significance.")
        print("Std. Err.: Standard error of the coefficient estimate.")
        print("t-value: Test statistic for the coefficient.")
        print("[0.025, 0.975]: 95% Confidence Interval for the coefficient.")
        print("F-statistic: Tests the overall significance of the model. A low p-value for F-statistic means the model is useful.")

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
        
        # Add R-squared to the plot
        r2_text = f"R² = {model_untransformed.rsquared:.3f}"
        ax.text2D(0.05, 0.95, r2_text, transform=ax.transAxes, fontsize=14,
                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                  verticalalignment='top') # Use ax.transAxes for relative positioning

        # Add a legend
        ax.legend()

        plt.tight_layout()
        
        # Save the plot
        plot_filename_3d = "log_likes_by_log_view_count_and_comment_count.png"
        plt.savefig(plot_filename_3d, dpi=300)
        print(f"Saved 3D plot: {plot_filename_3d}")
        plt.close(fig) # Close the plot to free memory

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    main()
