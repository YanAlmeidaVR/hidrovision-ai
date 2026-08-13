import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ana_horario.csv", parse_dates=["datahora"])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["datahora"], df["nivel_cm"], lw=0.6, color="#0B7FAB")

# quantis como referência de limiar
q = df["nivel_cm"].quantile([0.90, 0.95, 0.99])
for p, cor, nome in [(0.90, "orange", "P90"), (0.95, "orangered", "P95"), (0.99, "darkred", "P99")]:
    ax.axhline(q[p], color=cor, ls="--", lw=1, label=f"{nome} = {q[p]:.0f} cm")

ax.set_ylabel("nível (cm)")
ax.set_title("Rio Sapucaí — estação 61305000 (2023–2026)")
ax.legend()
plt.tight_layout()
plt.savefig("nivel_historico.png", dpi=150)
plt.show()

print(df["nivel_cm"].describe())
print("\nQuantis:", q.to_dict())
print("\nTop 5 picos:")
print(df.nlargest(5, "nivel_cm")[["datahora", "nivel_cm"]].to_string(index=False))