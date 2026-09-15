import sqlite3
con = sqlite3.connect("hidrovision.db")

print("--- ultimas 40 leituras ---")
for r in con.execute("select ts, nivel_cm, fonte, chuva_mm from leituras order by ts desc limit 40"):
    print(r)

print("\n--- alertas de hoje ---")
for r in con.execute("select ts, tipo, estado, origem from alertas where ts like '2026-09-10%' order by ts"):
    print(r)

print("\n--- previsoes de hoje ---")
for r in con.execute("select ts, nivel_atual, prev_6h, prev_24h, chuva_total_mm from previsoes where ts like '2026-09-10%' order by ts"):
    print(r)