import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

FILE = "/mnt/user-data/uploads/Dataset.csv"
df   = pd.read_csv(FILE)

# ── SECTION 2: Basic Dataset Exploration ─────────────────────
print("=" * 62)
print("   NBA PLAYER DEMOGRAPHICS & SALARY ANALYSIS")
print("=" * 62)

print("\n──────────────────────────────────────────────────────────")
print(" SECTION 2 │ BASIC DATASET EXPLORATION")
print("──────────────────────────────────────────────────────────")

print("\n▸ First 5 records:")
print(df.head().to_string(index=True))

print("\n▸ Last 5 records:")
print(df.tail().to_string(index=True))

rows, cols = df.shape
print(f"\n▸ Dataset shape : {rows} rows × {cols} columns")

print("\n▸ Missing values per column:")
miss = df.isnull().sum().rename("Count")
miss_pct = (miss / rows * 100).round(2).rename("  %")
print(pd.concat([miss, miss_pct], axis=1).to_string())

print("\n▸ Data types:")
print(df.dtypes.rename("Dtype").to_string())

# ── SECTION 3: Age Analysis ───────────────────────────────────
print("\n──────────────────────────────────────────────────────────")
print(" SECTION 3 │ AGE ANALYSIS")
print("──────────────────────────────────────────────────────────")

avg_age = df['Age'].mean()
min_age = df['Age'].min()
max_age = df['Age'].max()

print(f"\n▸ Average age : {avg_age:.2f} yrs")
print(f"▸ Minimum age : {min_age:.0f} yrs")
print(f"▸ Maximum age : {max_age:.0f} yrs")

bins   = [17, 21, 25, 29, 33, 45]
labels = ['18–21', '22–25', '26–29', '30–33', '34+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

age_sal = (df.dropna(subset=['Salary'])
             .groupby('Age Group', observed=True)['Salary']
             .mean()
             .sort_values(ascending=False))

print("\n▸ Average salary by age group:")
for grp, sal in age_sal.items():
    marker = " ◀ HIGHEST" if grp == age_sal.idxmax() else ""
    print(f"   {grp}: ${sal:>12,.0f}{marker}")

# ── SECTION 4: Position-Based Salary Analysis ────────────────
print("\n──────────────────────────────────────────────────────────")
print(" SECTION 4 │ POSITION-BASED SALARY ANALYSIS")
print("──────────────────────────────────────────────────────────")

PNAMES = {'PG':'Point Guard','SG':'Shooting Guard',
          'SF':'Small Forward','PF':'Power Forward','C':'Center'}

pos_sal = (df.dropna(subset=['Salary','Position'])
             .groupby('Position')['Salary']
             .mean()
             .sort_values(ascending=False))

print("\n▸ Average salary by position:")
for pos, sal in pos_sal.items():
    marker = " ◀ HIGHEST" if pos == pos_sal.idxmax() else ""
    print(f"   {PNAMES.get(pos,pos):17s} ({pos}): ${sal:>10,.0f}{marker}")

# ── SECTION 5: Team-Based Analysis ───────────────────────────
print("\n──────────────────────────────────────────────────────────")
print(" SECTION 5 │ TEAM-BASED ANALYSIS")
print("──────────────────────────────────────────────────────────")

team_stats = (df.groupby('Team')
                .agg(Players=('Name','count'),
                     Total_Salary=('Salary','sum'),
                     Avg_Salary=('Salary','mean'))
                .sort_values('Total_Salary', ascending=False))

print("\n▸ All teams — players, total payroll, average salary:")
print(f"   {'Team':<32} {'Players':>7} {'Total Payroll':>16} {'Avg Salary':>13}")
print("   " + "─" * 70)
for team, row in team_stats.iterrows():
    marker = " ◀" if team == team_stats['Total_Salary'].idxmax() else ""
    print(f"   {team:<32} {int(row.Players):>7} "
          f"${row.Total_Salary:>14,.0f} ${row.Avg_Salary:>11,.0f}{marker}")

best_team = team_stats['Total_Salary'].idxmax()
print(f"\n▸ Highest payroll team : {best_team}"
      f" → ${team_stats.loc[best_team,'Total_Salary']:,.0f}")

# ── SECTION 6: College Contribution ──────────────────────────
print("\n──────────────────────────────────────────────────────────")
print(" SECTION 6 │ COLLEGE CONTRIBUTION")
print("──────────────────────────────────────────────────────────")

col_count = (df.dropna(subset=['College'])
               .groupby('College')['Name']
               .count()
               .sort_values(ascending=False))

col_sal = (df.dropna(subset=['College','Salary'])
             .groupby('College')['Salary']
             .mean()
             .sort_values(ascending=False))

print("\n▸ Top 10 colleges by player count:")
for college, cnt in col_count.head(10).items():
    print(f"   {college:<35}: {cnt} players")

print("\n▸ Top 10 colleges by average player salary:")
for college, sal in col_sal.head(10).items():
    print(f"   {college:<35}: ${sal:,.0f}")


# ── VISUALIZATIONS ────────────────────────────────────────────

BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
DIM_TEXT = "#8b949e"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#f78166"
YELLOW   = "#e3b341"
PURPLE   = "#bc8cff"
TEAL     = "#39d353"

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=12.5, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color=DIM_TEXT, fontsize=9.5, labelpad=8)
    ax.set_ylabel(ylabel, color=DIM_TEXT, fontsize=9.5, labelpad=8)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)

fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))
fig.patch.set_facecolor(BG)
fig.suptitle("NBA Player Demographics & Salary Dashboard",
             color=TEXT, fontsize=16, fontweight='bold', y=1.02)

# ── Plot 1 │ Age Distribution Histogram ──────────────────────
ax1   = axes[0]
ages  = df['Age'].dropna()
n, bin_edges, patches = ax1.hist(
    ages, bins=range(int(ages.min()), int(ages.max())+2),
    color=BLUE, edgecolor=BG, linewidth=0.7, rwidth=0.82)

peak_idx = int(np.argmax(n))
patches[peak_idx].set_facecolor(YELLOW)
patches[peak_idx].set_edgecolor(TEXT)
patches[peak_idx].set_linewidth(1.2)

ax1.axvline(avg_age, color=ORANGE, linewidth=1.8, linestyle='--',
            label=f'Mean  {avg_age:.1f} yrs')
ax1.axvline(ages.median(), color=TEAL, linewidth=1.8, linestyle=':',
            label=f'Median  {ages.median():.1f} yrs')
ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
           fontsize=8.5, framealpha=0.9)

style_ax(ax1, "Age Distribution of NBA Players",
         "Age (years)", "Number of Players")
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# ── Plot 2 │ Avg Salary by Position ──────────────────────────
ax2      = axes[1]
pos_list = pos_sal.index.tolist()
labels2  = [f"{PNAMES.get(p,p)}\n({p})" for p in pos_list]
bar_cols = [YELLOW if p == pos_sal.idxmax() else GREEN for p in pos_list]

bars2 = ax2.bar(labels2, pos_sal.values / 1e6,
                color=bar_cols, edgecolor=BG, linewidth=0.7, width=0.58, zorder=3)

for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.08,
             f'${h:.2f}M', ha='center', va='bottom',
             color=TEXT, fontsize=8, fontweight='bold')

style_ax(ax2, "Average Salary by Position",
         "Position", "Average Salary (USD Millions)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax2.tick_params(axis='x', labelsize=8, rotation=0)

# ── Plot 3 │ Total Payroll by Team (Top 15) ───────────────────
ax3  = axes[2]
top15 = team_stats.head(15)['Total_Salary'].sort_values()
bar_cols3 = [YELLOW if t == best_team else PURPLE for t in top15.index]

bars3 = ax3.barh(top15.index, top15.values / 1e6,
                 color=bar_cols3, edgecolor=BG, linewidth=0.6,
                 height=0.68, zorder=3)

for bar in bars3:
    w = bar.get_width()
    ax3.text(w + 0.4, bar.get_y() + bar.get_height()/2,
             f'${w:.1f}M', va='center',
             color=TEXT, fontsize=7.5, fontweight='bold')

style_ax(ax3, "Total Payroll — Top 15 Teams",
         "Total Salary (USD Millions)", "")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax3.tick_params(axis='y', labelsize=8)
ax3.spines['bottom'].set_visible(True)
ax3.spines['bottom'].set_edgecolor(BORDER)
ax3.xaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.55, alpha=0.8)
ax3.yaxis.grid(False)

plt.tight_layout(pad=2.5)
OUT = "/mnt/user-data/outputs/nba_salary_dashboard.png"
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\n──────────────────────────────────────────────────────────")
print(f" Dashboard saved → {OUT}")
print("=" * 62)
print("   ANALYSIS COMPLETE")
print("=" * 62)
