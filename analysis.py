import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import date
from scipy.stats import pearsonr
from dotenv import load_dotenv
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        print("Successfully connected to the database for analysis.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL for analysis: {e}")
        print("Please check your database credentials and ensure PostgreSQL is running.")
        return None

def fetch_video_data_for_analysis(db_conn):
    """
    Fetches necessary video and channel data from the database for correlation analysis.
    Joins video and channel tables to get subscriber count.
    """
    query = sql.SQL("""
        SELECT
            v.like_count,
            v.published_at,
            v.view_count,
            v.comment_count,
            v.duration,
            v.tags,
            c.subscribers AS channel_subscribers
        FROM
            video AS v
        JOIN
            channel AS c ON v.channel_id = c.id
        WHERE
            v.like_count IS NOT NULL AND v.view_count IS NOT NULL
            AND v.comment_count IS NOT NULL AND v.duration IS NOT NULL
            AND c.subscribers IS NOT NULL
            AND v.like_count >= 0 AND v.view_count >= 0 AND v.comment_count >= 0 AND v.duration >= 0
            AND c.subscribers >= 0;
    """)
    
    try:
        df = pd.read_sql(query.as_string(db_conn), db_conn)
        print(f"Successfully fetched {len(df)} rows of data for analysis.")
        return df
    except Exception as e:
        print(f"Error fetching data for analysis: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def main():
    db_conn = None
    try:
        db_conn = get_db_connection()
        if not db_conn:
            return

        df = fetch_video_data_for_analysis(db_conn)

        if df.empty:
            print("No data available for analysis. Exiting.")
            return
        
        df['published_at'] = pd.to_datetime(df['published_at'])

        # 1. Video Age (in days)
        today = date.today()

        df['video_age_days'] = (pd.to_datetime(today) - df['published_at']).dt.days
        
        # Ensure age is not negative
        df['video_age_days'] = df['video_age_days'].apply(lambda x: max(0, x))

        # 2. Number of Tags
        df['num_tags'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # --- Prepare data for correlation and plotting ---
        analysis_df = df[[
            'like_count',
            'video_age_days',
            'view_count',
            'comment_count',
            'duration',
            'num_tags',
            'channel_subscribers'
        ]].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Convert relevant columns to numeric type
        for col in ['like_count', 'video_age_days', 'view_count', 'comment_count', 'duration', 'num_tags', 'channel_subscribers']:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
        
        # Drop rows with NaN values that might result from 'coerce' or initial fetch issues
        analysis_df.dropna(inplace=True)

        if analysis_df.empty:
            print("No valid numerical data after cleaning. Exiting.")
            return

        # --- Apply Log Transformation to highly skewed variables for better visualization and regression ---
        # Using log1p (log(1+x)) to handle potential zero values
        skewed_vars = ['like_count', 'view_count', 'comment_count', 'channel_subscribers']
        for var in skewed_vars:
            # Add a small constant (1) before log to handle zero values, then take log
            # Check if all values are zero before transforming, to avoid log(1) for all
            if not (analysis_df[var] == 0).all():
                analysis_df[f'log_{var}'] = np.log1p(analysis_df[var])
            else:
                analysis_df[f'log_{var}'] = 0 # If all zeros, keep it zero after log1p
                print(f"'{var}' contains only zero values, log1p transformation resulted in zeros.")

        print("\n--- Pearson Correlation Coefficients (R-value) with Like Count ---")
        
        # Independent variables for correlation and plotting
        independent_vars = [
            'video_age_days',
            'view_count',
            'comment_count',
            'duration',
            'num_tags',
            'channel_subscribers'
        ]
        
        # Prepare actual variables for plotting and correlation
        plotting_independent_vars = {
            'video_age_days': {'col': 'video_age_days', 'label': 'Tempo desde a Postagem (Dias)'},
            'view_count': {'col': 'log_view_count', 'label': 'Número de Visualizações (Log)'},
            'comment_count': {'col': 'log_comment_count', 'label': 'Número de Comentários (Log)'},
            'duration': {'col': 'duration', 'label': 'Duração do Vídeo (Segundos)'},
            'num_tags': {'col': 'num_tags', 'label': 'Número de Tags'},
            'channel_subscribers': {'col': 'log_channel_subscribers', 'label': 'Número de Inscritos do Canal (Log)'}
        }

        # Calculate correlations for original variables
        results = {}
        for var in independent_vars:
            if not analysis_df[var].std() == 0 and not analysis_df['like_count'].std() == 0:
                correlation, p_value = pearsonr(analysis_df['like_count'], analysis_df[var])
                results[var] = {'correlation': correlation, 'p_value': p_value}
                
                print(f"- {var.replace('_', ' ').title()}: R = {correlation:.4f} (p-value: {p_value:.4f})")
                
            else:
                print(f"- {var.replace('_', ' ').title()}: Correlation N/A (zero variance in data)")
                results[var] = {'correlation': np.nan, 'p_value': np.nan}

        # --- Generate and Save Correlation Graphs ---
        print("\n--- Generating Correlation Graphs ---")
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("crest")

        # Dependent variable for plotting
        y_var_plot = 'log_like_count'
        y_label_plot = 'Quantidade de Likes (Log)'

        for original_var, plot_info in plotting_independent_vars.items():
            x_var_plot = plot_info['col']
            x_label_plot = plot_info['label']

            # Skip plotting if the column doesn't exist or has zero variance
            if x_var_plot not in analysis_df.columns or analysis_df[original_var].std() == 0 or analysis_df[y_var_plot].std() == 0:
                print(f"Skipping plot for {original_var} due to missing data or zero variance in original data.")
                continue

            plt.figure(figsize=(10, 6))
            
            # Use sns.regplot for scatter plot with a linear regression line
            sns.regplot(x=analysis_df[x_var_plot], y=analysis_df[y_var_plot], scatter_kws={'alpha':0.3})
            
            # Get correlation for the *plotted* variables
            plot_correlation, _ = pearsonr(analysis_df[y_var_plot], analysis_df[x_var_plot])
            corr_text = f"R = {plot_correlation:.4f}"

            # Add annotations
            plt.title(f'{y_label_plot} vs. {x_label_plot}', fontsize=16)
            plt.xlabel(x_label_plot, fontsize=12)
            plt.ylabel(y_label_plot, fontsize=12)
            
            # Add correlation text to the plot
            plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

            plt.grid(True, linestyle='-', alpha=0.6)
            sns.despine()
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"log_likes_vs_{original_var}.png"
            plt.savefig(plot_filename, dpi=300)
            print(f"Saved plot: {plot_filename}")
            plt.close() # Close the plot to free memory

        # --- Save correlation results to a CSV file ---
        output_csv_filename = "correlation_results.csv"
        try:
            with open(output_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Variable', 'Correlation_Coefficient_R', 'P_Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for var, data in results.items():
                    writer.writerow({
                        'Variable': var.replace('_', ' ').title(),
                        'Correlation_Coefficient_R': f"{data['correlation']:.4f}",
                        'P_Value': f"{data['p_value']:.4f}" if not np.isnan(data['p_value']) else "N/A"
                    })
            print(f"\nCorrelation results saved to '{output_csv_filename}'")
        except IOError as e:
            print(f"Error saving correlation results to CSV file: {e}")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        if db_conn:
            db_conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    main()
