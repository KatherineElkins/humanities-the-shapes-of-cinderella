"""
The Shapes of Cinderella: Sentiment Analysis Visualization
Elkins, K. (2025). Humanities 14(10): 198.
https://doi.org/10.3390/h14100198

This script reproduces the sentiment analysis figures from the published paper.
Each figure shows clause-level sentiment with Savitzky-Golay smoothing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =============================================================================
# SENTIMENT DATA (Clause-level scores, -5 to +5)
# =============================================================================

# Ye Xian (112 clauses) - Tang Dynasty, c. 850 CE
ye_xian = np.array([
    0, 0, 0, -3, 1, 3, 2, 4, -5, -4, -4,  # Opening (1-11)
    2, 3, 2, 2, 0, 1, 3, 5, 4,  # Fish friendship (12-20)
    -2, -3, 2, -3, -2, -3, -2, -4,  # Discovery (21-28)
    -4, -5, -3, -2, -5, 1, -4, -3, -3,  # Murder (29-37)
    0, -4, -5,  # Grief (38-40)
    1, 3, 2, 2, -3, -2, 3, 4, 4,  # Divine intervention (41-49)
    2, 5,  # Magic works (50-51)
    2, 0, -3, 1, 2, 4, 4, -2, -3, -3, -2, -1, 0,  # Festival (52-64)
    0, 1, 1,  # Return (65-67)
    0, 0, 2, 2, 0, 1, 0, -1, 0, -1, 3, 3,  # Shoe journey (68-79)
    -2, -3, -1, 0, -1, -2,  # Investigation (80-85)
    0, 0, 2, 3, 4, 3, 5, 2, 4,  # Recognition (86-94)
    -3, 1, 0, -1, 1, 2,  # Stepfamily fate (95-100)
    2, 5, -3, -2, 3, -4, 1, 2, 2, -3, -2, -4  # Marriage & decline (101-112)
])

# Perrault's Cendrillon (257 clauses) - French, 1697
perrault = np.array([
    0, -3, -2, -2, 0, 4, 4,  # Opening (1-7)
    0, -3, -4, -4, -4, -3, -3, -3, -4, -2, -2, -1,  # Mistreatment (8-19)
    -3, -3, -3, -4,  # Father's powerlessness (20-23)
    0, -2, -3, -4, -1, -1, 1, 4, 3,  # Naming (24-32)
    3, 3, 2, 2, 3, 2, -3, -3, -2,  # Ball invitation (33-41)
    1, 2, 2, 0, 3, 3,  # Preparations (42-47)
    2, 2, 1, 3, 3, 2, 0,  # Getting ready (48-54)
    0, -1, -3, -2, -3, -5, 2, 3, 3,  # Cruel mockery (55-63)
    -1, 2, -1, -1, 0,  # Final preparations (64-68)
    3, 0, -2,  # Departure (69-71)
    -1, -3, 0, 1, -2, -3,  # Cinderella's tears (72-77)
    2, 2, -1, 3, 4,  # Fairy godmother (78-82)
    1, 1, 2, 0, 0,  # Transformation begins (83-87)
    0, 0, 2, 5,  # Pumpkin to coach (88-91)
    0, 0, 1, 3, 4, 4, 3,  # Mice to horses (92-98)
    0, 1, 1, 2, 0, 2, 3, 3,  # Rat to coachman (99-106)
    1, 0, 0, 3, 3, 4, 3,  # Lizards to footmen (107-113)
    3, 3, -1, -3,  # Key question (114-117)
    3, 5, 5, 5, 5,  # Dress transformation peak (118-122)
    4, 1, -1, -2, -2, -2, -3, 2,  # Warning (123-130)
    5, 3, 3, 4, 3,  # Arrival at ball (131-135)
    2, 2, 1, 4, 4, 5,  # The sensation (136-141)
    3, 2, 2, 2,  # Effect on others (142-145)
    4, 4, 5, 5,  # Dancing with prince (146-149)
    2, 3, 4, 2,  # Kindness to sisters (150-153)
    0, 2, -1,  # First departure (154-156)
    2, 3, 3,  # Return home (157-159)
    2, 0, 1, 0, 1, 1, 2,  # Sisters return (160-166)
    1, 2, 4, 3, 3,  # Sisters tell about ball (167-171)
    4, 0, 0, 3,  # Cinderella's questions (172-175)
    2, 1, -1, -1, -2, -3, -5, -4, 1, 2, 2,  # Request for dress (176-186)
    2, 3, 5, 4, -2, -3, -2, -1,  # Second ball (187-194)
    0, -1, 2, -2, -2, -3, -2, 1, 1,  # Chase and slipper (195-203)
    0, -1, -2, -3,  # Guards' report (204-207)
    0, 1, 0, 1, -1, 1, 3, 2, 3,  # Sisters report (208-216)
    0, 3, 4,  # Proclamation (217-219)
    2, 1, -1, 0, -2, -2,  # The search (220-225)
    0, 2, 3, -3, 0, 1, 3, 3, 3, 2, 2, 4, 4, 2, 4, 4,  # Recognition (226-241)
    3, 4, 5,  # Final transformation (242-244)
    3, 1, 2, 3, 4, 5, 5,  # Recognition and forgiveness (245-251)
    4, 5, 4, 4, 5, 4  # Happy ending (252-257)
])

# Grimm 1812 (275 clauses) - German, first edition
grimm_1812 = np.array([
    0, 3, 2, -3, -4, -1, -4, 2, 1, 3, 3, 3, 1, -5, -4, 2, 0, 1,  # Opening & death (1-18)
    0, 0, 1, -2, -1, 0, 1, -3, -1, -2, -4,  # Stepfamily (19-29)
    -3, -2, -3, -2, -3, -3, -2, -2, -3, -2, -3, -2, -2, -4, -3, -3, -3, -3, -3, -2, -2, -2,  # Abuse (30-51)
    2, 3, 2, 1, -1, -2, -2, 0, -1, 0, -3, -2, -2, -3, -2, -2, -3, -3, -2, -1, -2, -2, -3,  # Ball prep (52-74)
    -1, -2, -2, -3, -1, -3, -3, -3, -2, -3, -1, 0, 3, 2, 2, 4, 3, 2, 3, 3, 3, 4, 4, 3,  # First help (75-98)
    1, 0, 1, 1, 0, 2, -1, 2, 0, 0, -3, -2,  # (99-110)
    -1, -2, -1, 0, 1, 0, -2, -3, -2, -1, -2, -3, -2, -3, -2,  # Second prep (111-125)
    3, 2, 4, 3, 2, 3, 3, 3, 4, 1, 2, 3, 4, 4, 3, 4, 4, 3, 2, 1,  # Second help (126-145)
    3, 4, 4, 3, 4, 3, 2, 3, 3, 2, 1, 0, 0, -1, -2,  # Second ball (146-160)
    -1, -2, -2, -3, -2, -3, -2, -1, -2, -3, -2, -3, -2, -2, -1,  # Third prep (161-175)
    3, 2, 4, 3, 2, 3, 3, 4, 5, 5, 4, 5, 5, 4, 3, 4, 4, 3, 2, 3,  # Golden dress (176-195)
    4, 5, 5, 4, 3, 3, 2, 1, 0, -2, -3, -2, -1, 1, 2, 1, 0, -1, -2, -1,  # Ball & shoe (196-215)
    2, 3, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2,  # Shoe search (216-230)
    -2, -3, -3, -4, -3, -4, -3, -2, -1, 0, 1, -2, -4, -3, -2,  # First sister (231-245)
    -2, -3, -3, -4, -4, -4, -3, -2, -1, 0, 1, -2, -4, -3, -2,  # Second sister (246-260)
    -1, 0, 1, 2, 3, 2, 3, 4, 5, 5, 4, 5, 4, 3, 5  # Recognition (261-275)
])

# Grimm 1857 (315 clauses) - German, seventh edition
grimm_1857 = np.array([
    -2, -3, -2, 1, 3, 2, 2, -5, -2, -3, 1, 0, 1, -2,  # Opening & death (1-14)
    -1, 1, -3, -4, -3, -2, -3, -3, -2, -2, -3, -3, -2, -3, -2, -2, -4, -3, -3, -3, -2, -3, -3, -2, -2,  # Abuse (15-39)
    0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 2, 1, 0, 2, 1, 3, 1, 0, 3, 2, 4,  # Hazel branch NEW (40-62)
    2, 2, 1, 2, 1, 1, -1, -2, -2, 0, -1, -3, -3, -2, -3, -2, -3, -2, -1, -2, -2, 1,  # Ball prep (63-84)
    0, 2, 2, 2, 1, 3, 3, 4, 2, 2, 3, 3, 4, 1, 2, 3, 3, -3, -3, -3,  # First help (85-104)
    -3, -3, -3, 1, -2, 0, 2, 2, 2, 1, 3, 3, 4, 2, 2, 3, 3, 4, 1, -3,  # Second task (105-124)
    1, 1, 2, 3, 4, 4, 3, 3, 4, 4, 4, 3, 4, 4, 5, 4, 4, 4,  # First ball (125-142)
    3, 2, 1, 0, -1, 1, 2, 1, 0, -1, -2, -1, 0, 1, 1, 0, -1, -1,  # Escape (143-160)
    0, 1, 1, 2, 3, 4, 4, 3, 3, 4, 5, 4, 4, 4, 4,  # Second ball (161-175)
    3, 2, 1, 0, -1, 1, 2, 1, 0, -1, -2, -1, 0, 1, 0,  # Second escape (176-190)
    0, 1, 1, 2, 3, 4, 5, 5, 4, 4, 5, 5, 4, 4, 4,  # Third ball (191-205)
    3, 2, 1, 0, -1, -2, -1, 2, 3, 2, 1, 0, -1, -1, 0,  # Escape with shoe (206-220)
    2, 3, 2, 1, 2, 1, 0, -1, -2, -3, -4, -3, -2, -1, 0,  # Shoe test (221-235)
    -2, -3, -3, -4, -3, -2, -1, 0, 1, -1, -4, -3, -2, -1, 0,  # First sister (236-250)
    -2, -3, -3, -4, -4, -3, -2, -1, 0, 1, -1, -4, -3, -2, 0,  # Second sister (251-265)
    -1, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 3, 2, 3, 4, 4,  # Recognition (266-285)
    3, 4, 4, 3, 3, 2, 3, 3, 4, 4,  # Wedding (286-295)
    2, 1, 0, -1, -2, -3, -4, -3, -2, -3, -4, -3, -2, -1, 0,  # Eye pecking NEW (296-310)
    -2, -3, -2, -1, 0  # Final (311-315)
])

# =============================================================================
# SMOOTHING FUNCTION
# =============================================================================

def smooth_with_reflection(data, window_size=7):
    """
    Apply Savitzky-Golay smoothing with reflection padding at edges.
    This preserves peaks/valleys better than simple moving average.
    """
    if window_size % 2 == 0:
        window_size += 1
    if len(data) < window_size:
        return data
    
    # Reflection padding
    pad = window_size // 2
    padded = np.pad(data, pad, mode='reflect')
    
    # Savitzky-Golay filter (polynomial order 2)
    smoothed = savgol_filter(padded, window_size, 2)
    
    return smoothed[pad:-pad]

# =============================================================================
# FIGURE 1: YE XIAN
# =============================================================================

def plot_ye_xian():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(ye_xian) + 1)
    
    # Plot layers
    ax.plot(x, ye_xian, color='gray', alpha=0.4, lw=0.8, label='Raw clauses')
    ax.plot(x, smooth_with_reflection(ye_xian, 7), color='#2c5aa0', lw=2, label='Medium smooth (w=7)')
    ax.plot(x, smooth_with_reflection(ye_xian, 15), color='#8B0000', lw=2.5, label='Heavy smooth (w=15)')
    
    # Key annotations
    annotations = [
        (9, -5, "Father dies\n父卒"),
        (19, 5, "Fish pillows head\n魚必露首枕岸"),
        (30, -5, "Fish murdered\n斫殺之"),
        (51, 5, "Bones respond\n金璣衣食"),
        (92, 5, "Celestial beauty\n色若天人"),
        (112, -4, "Bones stop\n不復應"),
    ]
    
    for clause, sentiment, label in annotations:
        ax.annotate(label, xy=(clause, sentiment), 
                   xytext=(clause, sentiment + (2.5 if sentiment > 0 else -2.5)),
                   ha='center', fontsize=8, style='italic',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    
    ax.set_xlabel('Narrative Units', fontsize=11)
    ax.set_ylabel('Sentiment', fontsize=11)
    ax.set_title('Figure 1. Ye Xian (c. 850 CE): Recognition Through Sacred Reciprocity', 
                fontsize=12, style='italic')
    ax.set_ylim(-6, 6)
    ax.set_xlim(0, len(ye_xian) + 1)
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# FIGURE 2: PERRAULT
# =============================================================================

def plot_perrault():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(perrault) + 1)
    
    ax.plot(x, perrault, color='gray', alpha=0.4, lw=0.8, label='Raw clauses')
    ax.plot(x, smooth_with_reflection(perrault, 7), color='#228B22', lw=2, label='Medium smooth (w=7)')
    ax.plot(x, smooth_with_reflection(perrault, 15), color='#006400', lw=2.5, label='Heavy smooth (w=15)')
    
    annotations = [
        (27, -4, "Culcendron"),
        (60, -5, "on rirait bien"),
        (82, 4, "Fairy godmother"),
        (120, 5, "Dress transformation"),
        (148, 5, "Contemplation"),
        (182, -5, "vilain Culcendron"),
        (250, 5, "Forgiveness"),
    ]
    
    for clause, sentiment, label in annotations:
        ax.annotate(label, xy=(clause, sentiment),
                   xytext=(clause, sentiment + (2 if sentiment > 0 else -2)),
                   ha='center', fontsize=8, style='italic',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    
    ax.set_xlabel('Narrative Units', fontsize=11)
    ax.set_ylabel('Sentiment', fontsize=11)
    ax.set_title("Figure 2. Perrault's Cendrillon (1697): Linguistic Violence and Material Escape",
                fontsize=12, style='italic')
    ax.set_ylim(-6, 6)
    ax.set_xlim(0, len(perrault) + 1)
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# FIGURE 3: GRIMM 1812
# =============================================================================

def plot_grimm_1812():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(grimm_1812) + 1)
    
    ax.plot(x, grimm_1812, color='gray', alpha=0.4, lw=0.8, label='Raw clauses')
    ax.plot(x, smooth_with_reflection(grimm_1812, 7), color='#4169E1', lw=2, label='Medium smooth (w=7)')
    ax.plot(x, smooth_with_reflection(grimm_1812, 15), color='#8B4513', lw=2.5, label='Heavy smooth (w=15)')
    
    annotations = [
        (14, -5, "Mother dies\nverschied"),
        (91, 4, "Doves help\nTauben"),
        (140, 4, "Silver dress"),
        (184, 5, "Golden dress\nganz golden"),
        (197, 5, "Prince dances"),
        (234, -4, "Blood in shoe\nBlut im Schuck"),
        (269, 5, "Recognition\nerkannte er"),
    ]
    
    for clause, sentiment, label in annotations:
        ax.annotate(label, xy=(clause, sentiment),
                   xytext=(clause, sentiment + (2 if sentiment > 0 else -2)),
                   ha='center', fontsize=8, style='italic',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    
    ax.set_xlabel('Narrative Units', fontsize=11)
    ax.set_ylabel('Sentiment', fontsize=11)
    ax.set_title("Figure 3. Grimm's Aschenputtel (1812): Triadic Ascension Through Productive Suffering",
                fontsize=12, style='italic')
    ax.set_ylim(-6, 6)
    ax.set_xlim(0, len(grimm_1812) + 1)
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# FIGURE 4: GRIMM 1857
# =============================================================================

def plot_grimm_1857():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(grimm_1857) + 1)
    
    ax.plot(x, grimm_1857, color='gray', alpha=0.4, lw=0.8, label='Raw clauses')
    ax.plot(x, smooth_with_reflection(grimm_1857, 7), color='#9932CC', lw=2, label='Medium smooth (w=7)')
    ax.plot(x, smooth_with_reflection(grimm_1857, 15), color='#4B0082', lw=2.5, label='Heavy smooth (w=15)')
    
    annotations = [
        (5, 3, "God invoked\nder liebe Gott"),
        (8, -5, "Mother dies\nverschied"),
        (57, 3, "Tree grows\nschöner Baum"),
        (139, 5, "Dancing\ntanzte mit ihm"),
        (197, 5, "Golden shoes\nganz golden"),
        (276, 5, "Recognition\nerkannte er"),
        (302, -4, "Eyes pecked\nAuge aus"),
    ]
    
    for clause, sentiment, label in annotations:
        ax.annotate(label, xy=(clause, sentiment),
                   xytext=(clause, sentiment + (2 if sentiment > 0 else -2)),
                   ha='center', fontsize=8, style='italic',
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
    
    ax.set_xlabel('Narrative Units', fontsize=11)
    ax.set_ylabel('Sentiment', fontsize=11)
    ax.set_title("Figure 4. Grimm's Aschenputtel (1857): Christianization and Divine Retribution",
                fontsize=12, style='italic')
    ax.set_ylim(-6, 6)
    ax.set_xlim(0, len(grimm_1857) + 1)
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# FIGURE 5: COMPARATIVE (NORMALIZED)
# =============================================================================

def normalize_to_percent(data, n_points=100):
    """Normalize data to percentage of narrative progression."""
    from scipy.interpolate import interp1d
    x_orig = np.linspace(0, 100, len(data))
    f = interp1d(x_orig, data, kind='linear')
    return f(np.linspace(0, 100, n_points))

def plot_comparative():
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.linspace(0, 100, 100)
    
    # Normalize and smooth each version
    ye_xian_norm = smooth_with_reflection(normalize_to_percent(ye_xian), 7)
    perrault_norm = smooth_with_reflection(normalize_to_percent(perrault), 7)
    grimm_1812_norm = smooth_with_reflection(normalize_to_percent(grimm_1812), 7)
    grimm_1857_norm = smooth_with_reflection(normalize_to_percent(grimm_1857), 7)
    
    ax.plot(x, ye_xian_norm, color='#8B0000', lw=2, label='Ye Xian (c. 850 CE)')
    ax.plot(x, perrault_norm, color='#228B22', lw=2, label='Perrault (1697)')
    ax.plot(x, grimm_1812_norm, color='#4169E1', lw=2, label='Grimm (1812)')
    ax.plot(x, grimm_1857_norm, color='#9932CC', lw=2, label='Grimm (1857)')
    
    # Transformation zone
    ax.axvspan(41, 54, alpha=0.15, color='gold', label='Transformation zone (41-54%)')
    
    ax.set_xlabel('Narrative Progression (%)', fontsize=11)
    ax.set_ylabel('Sentiment', fontsize=11)
    ax.set_title('Figure 5. Comparative Sentiment Analysis: Four Cinderella Variants',
                fontsize=12, style='italic')
    ax.set_ylim(-5, 5)
    ax.set_xlim(0, 100)
    ax.axhline(0, color='black', lw=0.5)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN: Generate all figures
# =============================================================================

if __name__ == "__main__":
    print("Generating figures...")
    print(f"  Ye Xian: {len(ye_xian)} clauses")
    print(f"  Perrault: {len(perrault)} clauses")
    print(f"  Grimm 1812: {len(grimm_1812)} clauses")
    print(f"  Grimm 1857: {len(grimm_1857)} clauses")
    
    # Generate and save each figure
    fig1 = plot_ye_xian()
    fig1.savefig('figure1_ye_xian.png', dpi=300, bbox_inches='tight')
    print("  Saved: figure1_ye_xian.png")
    
    fig2 = plot_perrault()
    fig2.savefig('figure2_perrault.png', dpi=300, bbox_inches='tight')
    print("  Saved: figure2_perrault.png")
    
    fig3 = plot_grimm_1812()
    fig3.savefig('figure3_grimm_1812.png', dpi=300, bbox_inches='tight')
    print("  Saved: figure3_grimm_1812.png")
    
    fig4 = plot_grimm_1857()
    fig4.savefig('figure4_grimm_1857.png', dpi=300, bbox_inches='tight')
    print("  Saved: figure4_grimm_1857.png")
    
    fig5 = plot_comparative()
    fig5.savefig('figure5_comparative.png', dpi=300, bbox_inches='tight')
    print("  Saved: figure5_comparative.png")
    
    plt.show()
    print("\nDone!")
