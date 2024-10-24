import pathlib
import numpy as np
import re
import matplotlib.pyplot as plt

DICE = {
    "d4": lambda: np.random.randint(1, 5),
    "d6": lambda: np.random.randint(1, 7),
    "d8": lambda: np.random.randint(1, 9),
    "d10": lambda: np.random.randint(1, 11),
    "d12": lambda: np.random.randint(1, 13),
    "d20": lambda: np.random.randint(1, 21),
    "d100": lambda: np.random.randint(1, 101)
}

def roll_dice(dice_str: str, keep: str = None) -> int:
    m = re.match(r"(\d+)d(\d+)([+-]\d+)?", dice_str)
    if not m:
        raise ValueError("Invalid dice string %s" % dice_str)
    
    num_dice, dice_type, modifier = m.groups()
    num_dice = int(num_dice)
    dice_type = int(dice_type)
    modifier = int(modifier) if modifier else 0
    rolls = [DICE[f"d{dice_type}"]() for _ in range(num_dice)]
    
    if keep is None:
        return int(np.sum(rolls)) + modifier
    elif keep.startswith("highest"):
        keep_n = int(keep.split("_")[1])
        return int(np.sum(np.sort(rolls)[-keep_n:])) + modifier
    elif keep.startswith("lowest"):
        keep_n = int(keep.split("_")[1])
        return int(np.sum(np.sort(rolls)[:keep_n])) + modifier
    else:
        raise ValueError("Invalid keep method. Must be 'highest_{n}' or 'lowest_{n}'")

def _plot_distribution(rolls: list[int], savepath: str = None, title: str = None):
    plt.hist(rolls, bins=np.arange(min(rolls), max(rolls) + 2) - 0.5, density=True, rwidth=0.8)
    plt.xlabel("Roll")
    plt.ylabel("Frequency")
    plt.title(title)
    if savepath:
        savepath = pathlib.Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath)
    else:
        plt.show()

    plt.close()

def plot_distribution(dice_str: str,
                      keep: str = None,
                      num_rolls: int = 1000,
                      savepath: str = None,
                      title: str = None):
    if not re.match(r"^\d*d\d+$", dice_str):
        raise ValueError("Invalid dice string")
    if num_rolls < 1:
        raise ValueError("Invalid number of rolls")
    rolls = [roll_dice(dice_str, keep) for _ in range(num_rolls)]
    _plot_distribution(rolls, savepath, title)

def main():
    num_rolls = 10000
    for num_dice in range(1, 4):
        for dice_type in [4, 6, 8, 10, 12, 20, 100]:
            for advantage in ["advantage", "disadvantage", None]:
                dice_str = f"{num_dice}d{dice_type}"
                if advantage == "advantage":
                    rolls = [max([roll_dice(dice_str), roll_dice(dice_str)]) for _ in range(num_rolls)]
                    adv_str = "_advantage"
                    adv_title = " with Advantage"
                elif advantage == "disadvantage":
                    rolls = [min([roll_dice(dice_str), roll_dice(dice_str)]) for _ in range(num_rolls)]
                    adv_str = "_disadvantage"
                    adv_title = " with Disadvantage"
                else:
                    rolls = [roll_dice(dice_str) for _ in range(num_rolls)]
                    adv_str = ""
                    adv_title = ""
                savepath = f"dice_stats/d{dice_type}/{num_dice}d{dice_type}{adv_str}.png"
                title = f"{num_dice}d{dice_type}{adv_title}"
                _plot_distribution(rolls, savepath, title)

                # plot Survival Function (1 - CDF)
                plt.hist(rolls, bins=np.arange(min(rolls), max(rolls) + 2) - 0.5, density=True, rwidth=0.8, cumulative=-1)
                plt.xlabel("Roll")
                plt.ylabel("Survival Function")
                plt.title(title)
                plt.savefig(savepath.replace(".png", "_sf.png"))
                plt.close()

                # Plot CDF
                plt.hist(rolls, bins=np.arange(min(rolls), max(rolls) + 2) - 0.5, density=True, rwidth=0.8, cumulative=True)
                plt.xlabel("Roll")
                plt.ylabel("CDF")
                plt.title(title)
                plt.savefig(savepath.replace(".png", "_cdf.png"))
                plt.close()


    # Plot Ability Score Distribution
    plot_distribution(
        dice_str="4d6",
        keep="highest_3",
        num_rolls=num_rolls,
        savepath="dice_stats/ability_scores.png",
        title="Ability Score Distribution"
    )

    # Some interesting statistics
    lines = []

    lines.append(f"E1: Event that at least one ability score is an 8 or lower")
    pE1 = np.mean([
        min(roll_dice("4d6", keep="highest_3") for _ in range(4)) <= 8
        for _ in range(num_rolls)
    ])
    lines.append(f"P(E1) = {pE1:.2f}\n")

    lines.append(f"E2: Event that at least one ability score is an 16 or higher")
    pE2 = np.mean([
        max(roll_dice("4d6", keep="highest_3") for _ in range(4)) >= 16
        for _ in range(num_rolls)
    ])
    lines.append(f"P(E2) = {pE2:.2f}\n")

    lines.append(f"E3: Event that all ability scores are 12 or higher")
    pE3 = np.mean([
        all(roll_dice("4d6", keep="highest_3") >= 12 for _ in range(4))
        for _ in range(num_rolls)
    ])
    lines.append(f"P(E3) = {pE3:.2f}\n")

    lines.append(f"E4: Event that at least 2 ability scores are 14 or higher")
    pE4 = np.mean([
        sum(roll_dice("4d6", keep="highest_3") >= 14 for _ in range(4)) >= 2
        for _ in range(num_rolls)
    ])
    lines.append(f"P(E4) = {pE4:.2f}\n")

    stats_savepath = pathlib.Path("dice_stats/stats.txt")
    stats_savepath.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_savepath, "w") as f:
        f.write("\n".join(lines))

        
    

if __name__ == "__main__":
    main()