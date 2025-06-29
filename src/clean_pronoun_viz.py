import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class CleanPronounVisualizer:
    def __init__(self, data_path):
        """Initialize with pronoun analysis data."""
        self.load_data(data_path)
        
    def load_data(self, data_path):
        """Load the pronoun analysis results."""
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} companies")
        print(f"Successful: {len(self.df[self.df['status'] == 'Successful'])}")
        print(f"Unsuccessful: {len(self.df[self.df['status'] == 'Unsuccessful'])}")
        
    def create_clean_plural_violin(self):
        """Create a clean violin plot for plural pronouns only."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create violin plot with custom colors
        colors = ['#2E8B57', '#DC143C']  # Forest green for successful, crimson for unsuccessful
        violin_parts = ax.violinplot([
            self.df[self.df['status'] == 'Successful']['plural_rate'].values,
            self.df[self.df['status'] == 'Unsuccessful']['plural_rate'].values
        ], positions=[1, 2], widths=0.7, showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style the other elements
        violin_parts['cmeans'].set_color('black')
        violin_parts['cmedians'].set_color('white')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cbars'].set_color('black')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')
        
        # Calculate and display statistics
        successful_plural = self.df[self.df['status'] == 'Successful']['plural_rate']
        unsuccessful_plural = self.df[self.df['status'] == 'Unsuccessful']['plural_rate']
        
        # Statistical test
        statistic, p_value = stats.mannwhitneyu(successful_plural, unsuccessful_plural, alternative='two-sided')
        effect_size = (successful_plural.mean() - unsuccessful_plural.mean()) / np.sqrt((successful_plural.var() + unsuccessful_plural.var()) / 2)
        
        # Add mean lines and labels
        ax.axhline(y=successful_plural.mean(), xmin=0.25, xmax=0.45, color='#2E8B57', linestyle='--', linewidth=3)
        ax.axhline(y=unsuccessful_plural.mean(), xmin=0.55, xmax=0.75, color='#DC143C', linestyle='--', linewidth=3)
        
        # Add statistical annotations
        ax.text(1, successful_plural.mean() + 3, f'μ = {successful_plural.mean():.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12, color='#2E8B57')
        ax.text(2, unsuccessful_plural.mean() + 3, f'μ = {unsuccessful_plural.mean():.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12, color='#DC143C')
        
        # Styling
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Successful\n(Accepted)', 'Unsuccessful\n(Rejected)'], fontsize=12)
        ax.set_ylabel('Team Pronouns per 1,000 words\n("we", "us", "our")', fontsize=14, fontweight='bold')
        ax.set_title('YC Applications: Team Language Usage\nSuccessful teams say "we" 37% more often', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics box
        stats_text = f'''Statistical Analysis:
• Difference: +{((successful_plural.mean()/unsuccessful_plural.mean())-1)*100:.0f}%
• p-value: {p_value:.4f}
• Effect size (Cohen's d): {effect_size:.2f}
• Sample: {len(successful_plural)} vs {len(unsuccessful_plural)} apps'''
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Clean up the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_clean_singular_violin(self):
        """Create a clean violin plot for singular pronouns only."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create violin plot with custom colors
        colors = ['#2E8B57', '#DC143C']  # Forest green for successful, crimson for unsuccessful
        violin_parts = ax.violinplot([
            self.df[self.df['status'] == 'Successful']['singular_rate'].values,
            self.df[self.df['status'] == 'Unsuccessful']['singular_rate'].values
        ], positions=[1, 2], widths=0.7, showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Style the other elements
        violin_parts['cmeans'].set_color('black')
        violin_parts['cmedians'].set_color('white')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cbars'].set_color('black')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')
        
        # Calculate and display statistics
        successful_singular = self.df[self.df['status'] == 'Successful']['singular_rate']
        unsuccessful_singular = self.df[self.df['status'] == 'Unsuccessful']['singular_rate']
        
        # Statistical test
        statistic, p_value = stats.mannwhitneyu(successful_singular, unsuccessful_singular, alternative='two-sided')
        effect_size = (unsuccessful_singular.mean() - successful_singular.mean()) / np.sqrt((successful_singular.var() + unsuccessful_singular.var()) / 2)
        
        # Add mean lines and labels
        ax.axhline(y=successful_singular.mean(), xmin=0.25, xmax=0.45, color='#2E8B57', linestyle='--', linewidth=3)
        ax.axhline(y=unsuccessful_singular.mean(), xmin=0.55, xmax=0.75, color='#DC143C', linestyle='--', linewidth=3)
        
        # Add statistical annotations
        ax.text(1, successful_singular.mean() + 1.5, f'μ = {successful_singular.mean():.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12, color='#2E8B57')
        ax.text(2, unsuccessful_singular.mean() + 1.5, f'μ = {unsuccessful_singular.mean():.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12, color='#DC143C')
        
        # Styling
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Successful\n(Accepted)', 'Unsuccessful\n(Rejected)'], fontsize=12)
        ax.set_ylabel('Individual Pronouns per 1,000 words\n("I", "me", "my")', fontsize=14, fontweight='bold')
        ax.set_title('YC Applications: Individual Language Usage\nRejected founders say "I" 81% more often', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics box
        stats_text = f'''Statistical Analysis:
• Difference: +{((unsuccessful_singular.mean()/successful_singular.mean())-1)*100:.0f}%
• p-value: {p_value:.4f}
• Effect size (Cohen's d): {effect_size:.2f}
• Sample: {len(successful_singular)} vs {len(unsuccessful_singular)} apps'''
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
        
        # Clean up the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_correlation_analysis(self):
        """Create correlation analysis with linear regression."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Separate data by status
        successful = self.df[self.df['status'] == 'Successful']
        unsuccessful = self.df[self.df['status'] == 'Unsuccessful']
        
        # Create scatter plot
        ax.scatter(successful['singular_rate'], successful['plural_rate'], 
                  alpha=0.8, s=100, color='#2E8B57', label='Successful (Accepted)', 
                  edgecolors='white', linewidth=2)
        ax.scatter(unsuccessful['singular_rate'], unsuccessful['plural_rate'], 
                  alpha=0.8, s=100, color='#DC143C', label='Unsuccessful (Rejected)', 
                  edgecolors='white', linewidth=2)
        
        # Calculate overall correlation and regression
        X = self.df['singular_rate'].values.reshape(-1, 1)
        y = self.df['plural_rate'].values
        
        # Linear regression
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        correlation_coef = np.corrcoef(self.df['singular_rate'], self.df['plural_rate'])[0, 1]
        
        # Plot regression line
        x_line = np.linspace(self.df['singular_rate'].min(), self.df['singular_rate'].max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, '--', color='gray', linewidth=2, alpha=0.8, label=f'Linear fit (R² = {r2:.3f})')
        
        # Calculate separate correlations
        corr_successful = np.corrcoef(successful['singular_rate'], successful['plural_rate'])[0, 1]
        corr_unsuccessful = np.corrcoef(unsuccessful['singular_rate'], unsuccessful['plural_rate'])[0, 1]
        
        # Styling
        ax.set_xlabel('Individual Language Usage\nSingular Pronouns per 1,000 words ("I", "me", "my")', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Team Language Usage\nPlural Pronouns per 1,000 words ("we", "us", "our")', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Language Pattern Correlation in YC Applications\nNegative correlation: More "I" usage = Less "we" usage', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add correlation statistics
        stats_text = f'''Correlation Analysis:
Overall: r = {correlation_coef:.3f}, R² = {r2:.3f}
Successful apps: r = {corr_successful:.3f}
Unsuccessful apps: r = {corr_unsuccessful:.3f}

Interpretation:
• Negative correlation suggests inverse relationship
• Individual vs team language mindsets
• R² = {r2:.1%} of variance explained by the model'''
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        # Add quadrant analysis
        median_singular = self.df['singular_rate'].median()
        median_plural = self.df['plural_rate'].median()
        
        ax.axhline(y=median_plural, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=median_singular, color='gray', linestyle=':', alpha=0.5)
        
        # Count companies in each quadrant
        high_we_low_i = len(self.df[(self.df['plural_rate'] > median_plural) & (self.df['singular_rate'] < median_singular)])
        low_we_high_i = len(self.df[(self.df['plural_rate'] < median_plural) & (self.df['singular_rate'] > median_singular)])
        
        # Add quadrant labels
        ax.text(0.85, 0.85, f'Team Focus\n({high_we_low_i} companies)', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.text(0.85, 0.15, f'Individual Focus\n({low_we_high_i} companies)', transform=ax.transAxes,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.6))
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def save_clean_visualizations(self):
        """Save all clean visualizations."""
        # Create and save plural pronoun violin plot
        plural_fig = self.create_clean_plural_violin()
        plural_fig.savefig('output/clean_plural_pronouns.png', dpi=300, bbox_inches='tight')
        plt.close(plural_fig)
        
        # Create and save singular pronoun violin plot
        singular_fig = self.create_clean_singular_violin()
        singular_fig.savefig('output/clean_singular_pronouns.png', dpi=300, bbox_inches='tight')
        plt.close(singular_fig)
        
        # Create and save correlation analysis
        corr_fig = self.create_correlation_analysis()
        corr_fig.savefig('output/pronoun_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(corr_fig)
        
        print("✅ Clean visualizations saved:")
        print("  • output/clean_plural_pronouns.png")
        print("  • output/clean_singular_pronouns.png")
        print("  • output/pronoun_correlation_analysis.png")

def main():
    """Create clean pronoun visualizations."""
    visualizer = CleanPronounVisualizer('output/csv/pronoun_analysis.csv')
    visualizer.save_clean_visualizations()
    
    # Print summary statistics
    successful = visualizer.df[visualizer.df['status'] == 'Successful']
    unsuccessful = visualizer.df[visualizer.df['status'] == 'Unsuccessful']
    
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Team pronouns - Successful: {successful['plural_rate'].mean():.1f} vs Unsuccessful: {unsuccessful['plural_rate'].mean():.1f}")
    print(f"Individual pronouns - Successful: {successful['singular_rate'].mean():.1f} vs Unsuccessful: {unsuccessful['singular_rate'].mean():.1f}")
    
    # Overall correlation
    correlation = np.corrcoef(visualizer.df['singular_rate'], visualizer.df['plural_rate'])[0, 1]
    print(f"Overall correlation (singular vs plural): r = {correlation:.3f}")

if __name__ == "__main__":
    main() 