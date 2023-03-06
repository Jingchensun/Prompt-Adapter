import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


save_dir = "main_curves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

path = "Results.xlsx"  # this is the excel file containing the results (like the one we released)
file = pd.read_excel(path, sheet_name="imcls_fewshot")

datasets = [
    "OxfordPets", "Flowers102", "FGVCAircraft", "DTD",
    "EuroSAT", "StanfordCars", "Food101", "SUN397",
    "Caltech101", "UCF101", "ImageNet"
]

shots = [1, 2, 4, 8, 16]

COLORS = {
    "zs": "C4",
    "linear": "C4",
    "CoOp": "C0",
    "UPT": "C2",
    "tip_adapter_f": "C1",
    "prompt_adapter_f": "C3",
    "tip_adapter": "C5",
    "prompt_adapter": "C6"
}
MS = 3
ALPHA = 1
plt.rcParams.update({"font.size": 12})

average = {
    "zs": 0.,
    "CoOp": np.array([0., 0., 0., 0., 0.]),
    "UPT": np.array([0., 0., 0., 0., 0.]),
    "tip_adapter_f": np.array([0., 0., 0., 0., 0.]),
    "prompt_adapter_f": np.array([0., 0., 0., 0., 0.]),
    "linear": np.array([0., 0., 0., 0., 0.]),
    "tip_adapter": np.array([0., 0., 0., 0., 0.]),
    "prompt_adapter": np.array([0., 0., 0., 0., 0.])

}

for dataset in datasets:
    print(f"Processing {dataset} ...")

    zs = file[dataset][0]

    CoOp = file[dataset][2:7]
    CoOp = [float(num) for num in CoOp]

    UPT = file[dataset][7:12]
    UPT = [float(num) for num in UPT]

    tip_adapter_f = file[dataset][12:17]
    tip_adapter_f = [float(num) for num in tip_adapter_f]

    prompt_adapter_f = file[dataset][17:22]
    prompt_adapter_f = [float(num) for num in prompt_adapter_f]

    linear = file[dataset][22:27]
    linear = [float(num) for num in linear]

    tip_adapter = file[dataset][27:32]
    tip_adapter = [float(num) for num in tip_adapter]

    prompt_adapter = file[dataset][32:37]
    prompt_adapter = [float(num) for num in prompt_adapter]

    

    average["zs"] += zs
    average["CoOp"] += np.array(CoOp)
    average["UPT"] += np.array(UPT)
    average["tip_adapter_f"] += np.array(tip_adapter_f)
    average["prompt_adapter_f"] += np.array(prompt_adapter_f)
    average["linear"] += np.array(linear)
    average["tip_adapter"] += np.array(tip_adapter)
    average["prompt_adapter"] += np.array(prompt_adapter)

    # Plot
    values = [zs]
    values += linear
    values += CoOp
    values += UPT
    values += tip_adapter_f
    values += prompt_adapter_f
    values += prompt_adapter
    values += tip_adapter


    val_min, val_max = min(values), max(values)
    diff = val_max - val_min
    val_bot = val_min - diff*0.05
    val_top = val_max + diff*0.05

    fig, ax = plt.subplots()
    ax.set_facecolor("#EBEBEB")

    ax.set_xticks([0] + shots)
    ax.set_xticklabels([0] + shots)
    ax.set_xlabel("Number of labeled training examples per class")
    ax.set_ylabel("Score (%)")
    ax.grid(axis="x", color="white", linewidth=1)
    ax.axhline(zs, color="white", linewidth=1)
    ax.set_title(dataset)
    ax.set_ylim(val_bot, val_top)

    ax.plot(
        0, zs,
        marker="*",
        markersize=MS*1.5,
        color=COLORS["zs"],
        alpha=ALPHA
    )
    ax.plot(
        shots, CoOp,
        marker="o",
        markersize=MS,
        color=COLORS["CoOp"],
        label="CoOp",
        alpha=ALPHA
    )
    ax.plot(
        shots, UPT,
        marker="o",
        markersize=MS,
        color=COLORS["UPT"],
        label="UPT",
        alpha=ALPHA
    )
    ax.plot(
        shots, tip_adapter,
        marker="o",
        markersize=MS,
        color=COLORS["tip_adapter"],
        label="Tip-Adapter",
        alpha=ALPHA
    )

    ax.plot(
        shots, prompt_adapter,
        marker="o",
        markersize=MS,
        color=COLORS["prompt_adapter"],
        label="Prompt-Adapter",
        alpha=ALPHA
    )
    ax.plot(
        shots, tip_adapter_f,
        marker="o",
        markersize=MS,
        color=COLORS["tip_adapter_f"],
        label="Tip-Adapter-F",
        alpha=ALPHA
    )
    ax.plot(
        shots, prompt_adapter_f,
        marker="o",
        markersize=MS,
        color=COLORS["prompt_adapter_f"],
        label="Prompt-Adapter-F",
        alpha=ALPHA
    )
    ax.plot(
        shots, linear,
        marker="o",
        markersize=MS,
        color=COLORS["linear"],
        label="Linear Probe CLIP",
        linestyle="dotted",
        alpha=ALPHA
    )

    ax.text(-0.5, zs-diff*0.11, "Zero-shot\nCLIP", color=COLORS["zs"])
    ax.legend(loc="lower right")

    fig.savefig(f"{save_dir}/{dataset}.pdf", bbox_inches="tight")


# Plot
average = {k: v/len(datasets) for k, v in average.items()}
zs = average["zs"]
linear = list(average["linear"])
CoOp = list(average["CoOp"])
UPT = list(average["UPT"])
tip_adapter_f = list(average["tip_adapter_f"])
prompt_adapter_f = list(average["prompt_adapter_f"])
tip_adapter = list(average["tip_adapter"])
prompt_adapter = list(average["prompt_adapter"])


values = [zs]
values += linear
values += CoOp
values += UPT
values += tip_adapter_f
values += prompt_adapter_f
values += tip_adapter
values += prompt_adapter

val_min, val_max = min(values), max(values)
diff = val_max - val_min
val_bot = val_min - diff*0.05
val_top = val_max + diff*0.05

fig, ax = plt.subplots()
ax.set_facecolor("#EBEBEB")

ax.set_xticks([0] + shots)
ax.set_xticklabels([0] + shots)
ax.set_xlabel("Number of labeled training examples per class")
ax.set_ylabel("Score (%)")
ax.grid(axis="x", color="white", linewidth=1)
ax.axhline(zs, color="white", linewidth=1)
ax.set_title("Average over 11 datasets", fontweight="bold")
ax.set_ylim(val_bot, val_top)

ax.plot(
    0, zs,
    marker="*",
    markersize=MS*1.5,
    color=COLORS["zs"],
    alpha=ALPHA
)
ax.plot(
    shots, CoOp,
    marker="o",
    markersize=MS,
    color=COLORS["CoOp"],
    label="CoOp",
    alpha=ALPHA
)
ax.plot(
    shots, UPT,
    marker="o",
    markersize=MS,
    color=COLORS["UPT"],
    label="UPT",
    alpha=ALPHA
)
ax.plot(
        shots, tip_adapter,
        marker="o",
        markersize=MS,
        color=COLORS["tip_adapter"],
        label="Tip-Adapter",
        alpha=ALPHA
    )

ax.plot(
        shots, prompt_adapter,
        marker="o",
        markersize=MS,
        color=COLORS["prompt_adapter"],
        label="Prompt-Adapter",
        alpha=ALPHA
    )
ax.plot(
    shots, tip_adapter_f,
    marker="o",
    markersize=MS,
    color=COLORS["tip_adapter_f"],
    label="Tip-Adapter-F",
    alpha=ALPHA
)
ax.plot(
    shots, prompt_adapter_f,
    marker="o",
    markersize=MS,
    color=COLORS["prompt_adapter_f"],
    label="Prompt-Adapter-F",
    alpha=ALPHA
)
ax.plot(
    shots, linear,
    marker="o",
    markersize=MS,
    color=COLORS["linear"],
    label="Linear Probe CLIP",
    linestyle="dotted",
    alpha=ALPHA
)

ax.text(-0.5, zs-diff*0.11, "Zero-shot\nCLIP", color=COLORS["zs"])
ax.legend(loc="lower right")

fig.savefig(f"{save_dir}/average.pdf", bbox_inches="tight")
