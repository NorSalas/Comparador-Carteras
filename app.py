import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador 5 Carteras PRO v2.8.1", layout="wide", page_icon="📈")

COLS_ACTIVOS = ["MSCI World", "EM (Emerg)", "Small Caps", "Oro",
                "Bonos Corto", "Bonos Medio", "Bonos Largo", "REITs"]

# ========================================
# FUNCIONES BÁSICAS
# ========================================
def cagr(r):
    return np.prod(1 + r) ** (1 / len(r)) - 1 if len(r) > 0 else 0

def volatilidad(r):
    return np.std(r, ddof=1) if len(r) > 1 else 0

def sharpe(r):
    vol = volatilidad(r)
    return np.mean(r) / vol if vol > 0 else 0

def max_drawdown(r):
    curva = np.cumprod(1 + r)
    return (curva / np.maximum.accumulate(curva) - 1).min()

def ulcer_index(r):
    if len(r) == 0:
        return 0
    curva = np.cumprod(1 + r)
    max_prev = np.maximum.accumulate(curva)
    dd = (curva - max_prev) / max_prev * 100
    return np.sqrt(np.mean(dd**2))

def max_recovery_years(r):
    if len(r) == 0:
        return 0
    curva = np.cumprod(1 + r)
    max_prev = np.maximum.accumulate(curva)
    dd = (curva - max_prev) / max_prev
    recoveries = []
    for i in range(1, len(curva)):
        if dd[i] < -0.05:
            for j in range(i, len(curva)):
                if curva[j] >= max_prev[i]:
                    recoveries.append(j - i)
                    break
    return max(recoveries) if recoveries else 0

def rentabilidades_cartera(df, pesos):
    return df[COLS_ACTIVOS].values @ pesos

def rentabilidad_real(r_nominal, inflacion):
    return (1 + r_nominal) / (1 + inflacion) - 1

def simular_progresion(r_anual, horizonte, inicial, mensual):
    n_meses = horizonte * 12
    if len(r_anual) >= horizonte:
        r_usar = r_anual[-horizonte:]
    else:
        r_usar = np.tile(r_anual, (horizonte // len(r_anual)) + 1)[:horizonte]

    r_mensual = np.repeat((1 + r_usar) ** (1 / 12) - 1, 12)[:n_meses]

    valor = np.zeros(n_meses + 1)
    valor[0] = inicial
    for i in range(1, n_meses + 1):
        valor[i] = valor[i - 1] * (1 + r_mensual[i - 1]) + mensual
    return valor

def deflactar_curva(curva_nominal, inflacion_anual):
    n_meses = len(curva_nominal) - 1
    inflacion_mensual = np.repeat((1 + inflacion_anual) ** (1 / 12), 12)
    inflacion_mensual = inflacion_mensual[:n_meses]

    indice_precios = np.ones(n_meses + 1)
    for i in range(1, n_meses + 1):
        indice_precios[i] = indice_precios[i - 1] * inflacion_mensual[i - 1]

    return curva_nominal / indice_precios

def serie_drawdown(r):
    curva = np.cumprod(1 + r)
    max_prev = np.maximum.accumulate(curva)
    dd = curva / max_prev - 1
    return curva, dd

def episodios_drawdown(r, anos):
    curva = np.cumprod(1 + r)
    max_prev = np.maximum.accumulate(curva)
    dd = curva / max_prev - 1

    episodios = []
    i = 0
    n = len(dd)

    while i < n:
        if dd[i] < 0:
            inicio = i - 1 if i > 0 else i
            j = i
            valle = i

            while j < n and dd[j] < 0:
                if dd[j] < dd[valle]:
                    valle = j
                j += 1

            fin_dd = j - 1
            recuperado = j < n and dd[j] >= 0
            rec_anos = (j - inicio) if recuperado else np.nan

            episodios.append({
                "Inicio": int(anos[inicio]),
                "Valle": int(anos[valle]),
                "Fin DD": int(anos[fin_dd]),
                "Drawdown": dd[valle],
                "Recuperación (años)": rec_anos
            })
            i = j
        else:
            i += 1

    return pd.DataFrame(episodios)

def sensibilidad_inicio(df_hist_full, carteras_pesos, ventana):
    filas = []
    if len(df_hist_full) < ventana:
        return pd.DataFrame()

    for nombre, pesos in carteras_pesos.items():
        pesos = np.array(pesos)
        for i in range(0, len(df_hist_full) - ventana + 1):
            sub = df_hist_full.iloc[i:i + ventana].copy()
            anos = sub["Año"].values
            r_nom = rentabilidades_cartera(sub, pesos)

            filas.append({
                "Cartera": nombre,
                "Inicio": int(anos[0]),
                "Fin": int(anos[-1]),
                "CAGR": cagr(r_nom)
            })

    return pd.DataFrame(filas)

def heatmap_cagr(df_periodo, pesos):
    pesos = np.array(pesos)
    anos = df_periodo["Año"].values
    n = len(df_periodo)

    mat = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            sub = df_periodo.iloc[i:j+1]
            if len(sub) >= 2:
                r_nom = rentabilidades_cartera(sub, pesos)
                mat[i, j] = cagr(r_nom)

    return anos, mat

def scatter_drawdown_recovery(df_hist_periodo, carteras_pesos):
    filas = []
    anos = df_hist_periodo["Año"].values

    for nombre, pesos in carteras_pesos.items():
        r = rentabilidades_cartera(df_hist_periodo, np.array(pesos))
        epis = episodios_drawdown(r, anos)
        if len(epis) > 0:
            epis = epis.copy()
            epis["Cartera"] = nombre
            filas.append(epis)

    if len(filas) == 0:
        return pd.DataFrame()

    return pd.concat(filas, ignore_index=True)

# ========================================
# INTERÉS COMPUESTO
# ========================================
def simular_interes_compuesto(aporte_inicial, aporte_mensual, anos, interes_anual):
    n_meses = int(anos * 12)
    r_mensual = (1 + interes_anual) ** (1 / 12) - 1

    meses = np.arange(n_meses + 1)
    anos_x = meses / 12

    saldo = np.zeros(n_meses + 1)
    principal_inicial = np.zeros(n_meses + 1)
    aportaciones = np.zeros(n_meses + 1)
    intereses = np.zeros(n_meses + 1)

    saldo[0] = aporte_inicial
    principal_inicial[:] = aporte_inicial
    aportaciones[0] = 0
    intereses[0] = 0

    for m in range(1, n_meses + 1):
        saldo[m] = saldo[m - 1] * (1 + r_mensual) + aporte_mensual
        aportaciones[m] = aporte_mensual * m
        intereses[m] = saldo[m] - principal_inicial[m] - aportaciones[m]

    df = pd.DataFrame({
        "Mes": meses,
        "Años": anos_x,
        "Saldo": saldo,
        "Inicial": principal_inicial,
        "Aportaciones": aportaciones,
        "Intereses": intereses
    })

    return df

def formatear_tiempo_meses(meses_total):
    if pd.isna(meses_total):
        return "No alcanzado"
    meses_total = int(meses_total)
    anos = meses_total // 12
    meses_rest = meses_total % 12
    return f"{anos} años y {meses_rest} meses"

def tabla_hitos_compuesto(df, hitos=None):
    if hitos is None:
        hitos = np.arange(100000, 1000001, 100000)

    filas = []
    mes_hito_previo = 0

    for hito in hitos:
        sub = df[df["Saldo"] >= hito]

        if len(sub) == 0:
            filas.append({
                "Hito": hito,
                "Años": np.nan,
                "Meses": np.nan,
                "Tiempo": "No alcanzado",
                "Desde hito anterior": "No alcanzado",
                "Inicial": np.nan,
                "Aportaciones": np.nan,
                "Intereses": np.nan
            })
        else:
            fila = sub.iloc[0]
            meses_total = int(fila["Mes"])
            meses_tramo = meses_total - mes_hito_previo

            filas.append({
                "Hito": hito,
                "Años": meses_total / 12,
                "Meses": meses_total,
                "Tiempo": formatear_tiempo_meses(meses_total),
                "Desde hito anterior": formatear_tiempo_meses(meses_tramo),
                "Inicial": fila["Inicial"],
                "Aportaciones": fila["Aportaciones"],
                "Intereses": fila["Intereses"]
            })

            mes_hito_previo = meses_total

    return pd.DataFrame(filas)

# ========================================
# BACKTEST
# ========================================
def calcular_metricas(df_periodo, carteras_pesos, inicial, mensual):
    resultados = []
    progresiones_nom = {}
    progresiones_real = {}
    n_anos = len(df_periodo)

    inflacion = df_periodo["Inflación"].values
    aportado_total = inicial + mensual * 12 * n_anos

    for nombre, pesos in carteras_pesos.items():
        r_nom = rentabilidades_cartera(df_periodo, np.array(pesos))
        r_real = rentabilidad_real(r_nom, inflacion)

        curva_nom = simular_progresion(r_nom, n_anos, inicial, mensual)
        curva_real = deflactar_curva(curva_nom, inflacion)

        valor_final_nom = curva_nom[-1]
        valor_final_real = curva_real[-1]

        resultados.append({
            "Cartera": nombre,
            "CAGR": cagr(r_nom),
            "CAGR Real": cagr(r_real),
            "Vol": volatilidad(r_nom),
            "Sharpe": sharpe(r_nom),
            "MaxDD": max_drawdown(r_nom),
            "Ulcer": ulcer_index(r_nom),
            "Recovery": max_recovery_years(r_nom),
            "Aportado€": aportado_total / 1000,
            "Final Nominal€": valor_final_nom / 1000,
            "Final Real€": valor_final_real / 1000,
            "Ganancia Nom€": (valor_final_nom - aportado_total) / 1000,
            "Ganancia Real€": (valor_final_real - aportado_total) / 1000
        })

        progresiones_nom[nombre] = curva_nom
        progresiones_real[nombre] = curva_real

    df_res = pd.DataFrame(resultados).set_index("Cartera")
    return df_res, progresiones_nom, progresiones_real

# ========================================
# MONTECARLO VECTORIAL
# ========================================
def simular_progresion_mc(r_anual_mat, horizonte, inicial, mensual):
    # r_anual_mat: shape (n_simul, horizonte)
    n_simul, horizonte = r_anual_mat.shape
    n_meses = horizonte * 12

    r_mensual = (1 + r_anual_mat) ** (1 / 12) - 1       # (n_simul, horizonte)
    r_mensual = np.repeat(r_mensual, 12, axis=1)        # (n_simul, n_meses)

    valores = np.zeros((n_simul, n_meses + 1))
    valores[:, 0] = inicial

    for m in range(1, n_meses + 1):
        valores[:, m] = valores[:, m - 1] * (1 + r_mensual[:, m - 1]) + mensual

    return valores

def calcular_montecarlo(df_periodo, carteras_pesos, horizonte, inicial, mensual, n_simul=10000):
    resultados = []
    mc_trayectorias_nom = {}
    mc_trayectorias_real = {}

    r_activos = df_periodo[COLS_ACTIVOS].values
    medias_activos = np.mean(r_activos, axis=0)
    cov_activos = np.cov(r_activos, rowvar=False, ddof=1)

    inflacion_media = float(df_periodo["Inflación"].mean())

    for nombre, pesos in carteras_pesos.items():
        w = np.array(pesos)

        mu_c = float(medias_activos @ w)
        sigma_c = float(np.sqrt(w @ cov_activos @ w))

        # Generar retornos anuales simulados de golpe
        r_anual_sim = np.random.normal(mu_c, sigma_c, size=(n_simul, horizonte))

        # Inflación constante media (vectorizada)
        inflacion_sim = np.full((n_simul, horizonte), inflacion_media)
        r_real_sim = rentabilidad_real(r_anual_sim, inflacion_sim)

        # Métricas de rentabilidad y riesgo (vectorizadas)
        cagr_sim_nom = (np.prod(1 + r_anual_sim, axis=1) ** (1 / horizonte) - 1)
        cagr_sim_real = (np.prod(1 + r_real_sim, axis=1) ** (1 / horizonte) - 1)
        vol_sim = np.std(r_anual_sim, axis=1, ddof=1)

        # Simular progresión mensual y pasar a anual
        curvas_nom = simular_progresion_mc(r_anual_sim, horizonte, inicial, mensual)  # (n_simul, n_meses+1)
        n_meses = curvas_nom.shape[1] - 1
        indices_anuales = np.linspace(0, n_meses, horizonte + 1).astype(int)

        curvas_anuales_nom = curvas_nom[:, indices_anuales]

        # Deflactar (usando inflacion_media aproximada)
        inflacion_mensual = (1 + inflacion_media) ** (1 / 12) - 1
        indice_precios = np.cumprod(np.ones(n_meses + 1) * (1 + inflacion_mensual))
        curvas_reales = curvas_nom / indice_precios  # broadcasting
        curvas_anuales_real = curvas_reales[:, indices_anuales]

        finales_nom = curvas_anuales_nom[:, -1]
        finales_real = curvas_anuales_real[:, -1]

        p5_nom, p50_nom, p95_nom = np.percentile(finales_nom / 1000, [5, 50, 95])
        p5_real, p50_real, p95_real = np.percentile(finales_real / 1000, [5, 50, 95])

        # Percentiles por año para fan chart (en miles)
        p5_path_nom = np.percentile(curvas_anuales_nom / 1000.0, 5, axis=0)
        p50_path_nom = np.percentile(curvas_anuales_nom / 1000.0, 50, axis=0)
        p95_path_nom = np.percentile(curvas_anuales_nom / 1000.0, 95, axis=0)

        resultados.append({
            "Cartera": nombre,
            "CAGR": float(np.mean(cagr_sim_nom)),
            "CAGR Real": float(np.mean(cagr_sim_real)),
            "Vol": float(np.mean(vol_sim)),
            "Sharpe": float(np.mean(cagr_sim_nom) / np.mean(vol_sim)) if np.mean(vol_sim) > 0 else 0,
            "Ulcer": ulcer_index(r_activos @ w),
            "Recovery": max_recovery_years(r_activos @ w),
            "P5% Nom": float(p5_nom),
            "Mediana Nom": float(p50_nom),
            "P95% Nom": float(p95_nom),
            "P5% Real": float(p5_real),
            "Mediana Real": float(p50_real),
            "P95% Real": float(p95_real)
        })

        mc_trayectorias_nom[nombre] = {
            "finales": finales_nom / 1000.0,
            "p5": p5_path_nom,
            "p50": p50_path_nom,
            "p95": p95_path_nom
        }

        mc_trayectorias_real[nombre] = {
            "finales": finales_real / 1000.0,
            "p5": np.percentile(curvas_anuales_real / 1000.0, 5, axis=0),
            "p50": np.percentile(curvas_anuales_real / 1000.0, 50, axis=0),
            "p95": np.percentile(curvas_anuales_real / 1000.0, 95, axis=0)
        }

    df_res = pd.DataFrame(resultados).set_index("Cartera")
    return df_res, mc_trayectorias_nom, mc_trayectorias_real

# ========================================
# CARGA DATOS
# ========================================
@st.cache_data
def load_data():
    xls = pd.ExcelFile("simulador.xlsx")
    df_hist = pd.read_excel(xls, sheet_name="HistoricoCompleto", decimal=",")
    df_trabajo = pd.read_excel(xls, sheet_name="HistoricoTrabajo", decimal=",")
    df_carts = pd.read_excel(xls, sheet_name="Carteras", decimal=",")

    for df in [df_hist, df_trabajo]:
        cols_a_convertir = [c for c in COLS_ACTIVOS if c in df.columns]
        if "Inflación" in df.columns:
            cols_a_convertir = cols_a_convertir + ["Inflación"]

        if len(cols_a_convertir) > 0 and df[cols_a_convertir].abs().max().max() > 2:
            df[cols_a_convertir] = df[cols_a_convertir] / 100

    return df_hist, df_trabajo, df_carts

st.title("🏆 SIMULADOR 5 CARTERAS PRO v2.8.1 - ANÁLISIS VISUAL + INTERÉS COMPUESTO")
st.markdown("---")

df_hist, df_trabajo, df_carts = load_data()

if "Inflación" not in df_hist.columns:
    st.error("❌ La hoja HistoricoCompleto debe incluir una columna llamada 'Inflación'")
    st.stop()

# ========================================
# CONTROLES LATERALES
# ========================================
col1, col2 = st.columns([1, 4])

with col1:
    st.header("⚙️ PARÁMETROS")
    anio_min, anio_max = int(df_hist["Año"].min()), int(df_hist["Año"].max())
    anio_inicio = st.slider("📅 Año inicio", anio_min, anio_max - 10, 1990)
    anio_fin = st.slider("📅 Año fin", anio_inicio + 5, anio_max, 2025)
    horizonte = st.slider("⏰ Horizonte (años)", 15, 40, 30)
    inicial = st.number_input("💰 Capital inicial", 5000, 50000, 10000, 1000)
    mensual = st.number_input("💸 Aportación mensual", 0, 2000, 300, 50)
    # Cambiamos valor por defecto a 1000 simulaciones
    n_simul = st.slider("🎲 Simulaciones", 1000, 25000, 1000, 1000)
    ventana_sens = st.selectbox("🗓️ Sensibilidad (años)", [5, 10, 15], index=1)

    st.markdown("---")
    st.header("🏦 CARTERA 1 - EDITABLE")

    c1_mw = st.slider("🌍 MSCI World", 0.0, 1.0, 0.45, 0.01)
    c1_em = st.slider("🌏 EM Emergentes", 0.0, 1.0 - c1_mw, 0.075, 0.005)
    c1_sc = st.slider("📈 Small Caps", 0.0, 1.0 - (c1_mw + c1_em), 0.075, 0.005)
    c1_oro = st.slider("🥇 Oro", 0.0, 1.0 - (c1_mw + c1_em + c1_sc), 0.20, 0.01)

    resto = round(1.0 - (c1_mw + c1_em + c1_sc + c1_oro), 10)

    if resto <= 0:
        c1_rf_corto = 0.0
        c1_rf_medio = 0.0
        c1_rf_largo = 0.0
        st.caption("Sin peso disponible para bonos.")
    else:
        c1_rf_corto = st.slider("📏 Bonos Corto", 0.0, float(resto), 0.0, 0.01)

        resto_2 = round(resto - c1_rf_corto, 10)
        if resto_2 <= 0:
            c1_rf_medio = 0.0
            c1_rf_largo = 0.0
        else:
            c1_rf_medio = st.slider("📐 Bonos Medio", 0.0, float(resto_2), 0.0, 0.01)

            resto_3 = round(resto_2 - c1_rf_medio, 10)
            if resto_3 <= 0:
                c1_rf_largo = 0.0
            else:
                c1_rf_largo = st.slider("📐 Bonos Largo", 0.0, float(resto_3), float(resto_3), 0.01)

    c1_reits = 1.0 - (c1_mw + c1_em + c1_sc + c1_oro + c1_rf_corto + c1_rf_medio + c1_rf_largo)

    st.metric("🏠 REITs (auto)", f"{c1_reits:.1%}")
    st.metric("✅ TOTAL", f"{c1_mw + c1_em + c1_sc + c1_oro + c1_rf_corto + c1_rf_medio + c1_rf_largo + c1_reits:.1%}")

# ========================================
# RESULTADOS PRINCIPALES
# ========================================
with col2:
    carteras_pesos = {}
    for _, row in df_carts.iterrows():
        nombre = row["Cartera"]
        pesos = [row[col] for col in COLS_ACTIVOS]
        carteras_pesos[nombre] = pesos

    carteras_pesos["Cartera 1"] = [c1_mw, c1_em, c1_sc, c1_oro,
                                   c1_rf_corto, c1_rf_medio, c1_rf_largo, c1_reits]

    df_periodo_hist = df_hist[(df_hist["Año"] >= anio_inicio) & (df_hist["Año"] <= anio_fin)].copy()

    if len(df_periodo_hist) < 3:
        st.error("❌ Selecciona más años")
        st.stop()

    with st.spinner("🚀 Calculando backtest + MonteCarlo..."):
        df_res_hist, progresiones_hist_nom, progresiones_hist_real = calcular_metricas(
            df_periodo_hist, carteras_pesos, inicial, mensual
        )

        df_res_mc, mc_trayectorias_nom, mc_trayectorias_real = calcular_montecarlo(
            df_periodo_hist, carteras_pesos, horizonte, inicial, mensual, n_simul
        )

    st.header("🏅 TOP CARTERAS")
    col1_, col2_, col3_, col4_ = st.columns(4)
    with col1_:
        st.metric("🥇 CAGR", df_res_hist["CAGR"].idxmax(), f"{df_res_hist['CAGR'].max():.2%}")
    with col2_:
        st.metric("📉 CAGR Real", df_res_hist["CAGR Real"].idxmax(), f"{df_res_hist['CAGR Real'].max():.2%}")
    with col3_:
        st.metric("💰 Final Real€", df_res_hist["Final Real€"].idxmax(), f"€{df_res_hist['Final Real€'].max():,.0f}K")
    with col4_:
        st.metric("😌 Ulcer", df_res_hist["Ulcer"].idxmin(), f"{df_res_hist['Ulcer'].min():.1f}")

    col_tab1, col_tab2 = st.columns(2)

    with col_tab1:
        st.markdown("### 📊 **BACKTEST**")
        st.dataframe(
            df_res_hist.style.format({
                "CAGR": "{:.2%}",
                "CAGR Real": "{:.2%}",
                "Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "MaxDD": "{:.2%}",
                "Ulcer": "{:.1f}",
                "Recovery": "{:.0f}",
                "Aportado€": "€{:.0f}K",
                "Final Nominal€": "€{:.0f}K",
                "Final Real€": "€{:.0f}K",
                "Ganancia Nom€": "€{:.0f}K",
                "Ganancia Real€": "€{:.0f}K"
            }).background_gradient(subset=["CAGR", "CAGR Real", "Final Real€"], cmap="RdYlGn"),
            use_container_width=True,
            height=255
        )

    with col_tab2:
        st.markdown("### 🎲 **MONTECARLO**")
        st.dataframe(
            df_res_mc.style.format({
                "CAGR": "{:.2%}",
                "CAGR Real": "{:.2%}",
                "Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Ulcer": "{:.1f}",
                "Recovery": "{:.0f}",
                "P5% Nom": "€{:.0f}K",
                "Mediana Nom": "€{:.0f}K",
                "P95% Nom": "€{:.0f}K",
                "P5% Real": "€{:.0f}K",
                "Mediana Real": "€{:.0f}K",
                "P95% Real": "€{:.0f}K"
            }).background_gradient(subset=["Mediana Real", "P95% Real"], cmap="RdYlGn"),
            use_container_width=True,
            height=255
        )

    st.markdown("### 🔎 **Selección de cartera para gráficos individuales**")
    cartera_sel = st.selectbox("Elige cartera", list(carteras_pesos.keys()), index=0)

    pesos_sel = carteras_pesos[cartera_sel]
    r_sel = rentabilidades_cartera(df_periodo_hist, np.array(pesos_sel))
    inflacion_sel = df_periodo_hist["Inflación"].values
    r_sel_real = rentabilidad_real(r_sel, inflacion_sel)
    anos = df_periodo_hist["Año"].values

    st.markdown(f"### 💰 **Evolución Backtest - {cartera_sel}**")
    curva_nom_sel = progresiones_hist_nom[cartera_sel] / 1000
    curva_real_sel = progresiones_hist_real[cartera_sel] / 1000
    anos_bt = np.arange(0, len(curva_nom_sel))

    fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
    ax_bt.plot(anos_bt, curva_nom_sel, lw=3, color="royalblue", label="Nominal")
    ax_bt.plot(anos_bt, curva_real_sel, lw=3, color="crimson", label="Real")
    ax_bt.fill_between(anos_bt, curva_real_sel, curva_nom_sel, color="lightgray", alpha=0.3)
    ax_bt.set_xlabel("Años desde inicio")
    ax_bt.set_ylabel("€ (miles)")
    ax_bt.set_title(f"Evolución nominal vs real - {cartera_sel}")
    ax_bt.grid(True, alpha=0.3)
    ax_bt.legend()
    st.pyplot(fig_bt)

    st.markdown(f"### 🎯 **MONTECARLO - {cartera_sel}**")
    col_graf1, col_graf2 = st.columns(2)

    tray_info_nom = mc_trayectorias_nom[cartera_sel]
    finales_sel = tray_info_nom["finales"]
    anos_mc = np.arange(0, horizonte + 1)

    with col_graf1:
        fig_hist, ax_hist = plt.subplots(figsize=(9, 5))

        p1, p5, p50, p95, p99 = np.percentile(finales_sel, [1, 5, 50, 95, 99])
        vals_plot = finales_sel[finales_sel <= p99]

        ax_hist.hist(vals_plot, bins=40, alpha=0.7, color="skyblue",
                     edgecolor="navy", density=True)
        ax_hist.axvline(p5, color="orange", lw=3, ls="--", label=f"P5%: €{p5:,.0f}K")
        ax_hist.axvline(p50, color="red", lw=3, label=f"Mediana: €{p50:,.0f}K")
        ax_hist.axvline(p95, color="orange", lw=3, ls="--", label=f"P95%: €{p95:,.0f}K")

        ax_hist.set_xlim(left=max(0, p1 * 0.9), right=p99 * 1.03)
        ticks_x = np.linspace(max(0, p1 * 0.9), p99 * 1.03, 8)
        ax_hist.set_xticks(ticks_x)
        ax_hist.tick_params(axis="x", rotation=45)

        ax_hist.set_xlabel("Valor final nominal (€ miles)")
        ax_hist.set_ylabel("Densidad")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_title(f"Histograma MonteCarlo - {cartera_sel} (recorte visual p99)")
        st.pyplot(fig_hist)

    with col_graf2:
        fig_fan, ax_fan = plt.subplots(figsize=(9, 5))
        p5_path = tray_info_nom["p5"]
        p50_path = tray_info_nom["p50"]
        p95_path = tray_info_nom["p95"]

        ax_fan.fill_between(anos_mc, p5_path, p95_path, alpha=0.3,
                            color="skyblue", label="P5%-P95%")
        ax_fan.fill_between(anos_mc, p5_path, p50_path, alpha=0.5,
                            color="lightblue", label="P5%-Mediana")
        ax_fan.plot(anos_mc, p50_path, "r-", linewidth=3, label="Mediana")
        ax_fan.plot(anos_mc, p5_path, "orange", ls="--", linewidth=2, label="P5%")
        ax_fan.plot(anos_mc, p95_path, "orange", ls="--", linewidth=2, label="P95%")

        ax_fan.set_ylabel("€ nominales (miles)")
        ax_fan.set_xlabel("Años")
        ax_fan.legend()
        ax_fan.grid(True, alpha=0.3)
        ax_fan.set_title(f"Fan chart MonteCarlo - {cartera_sel}")
        st.pyplot(fig_fan)

    st.markdown(f"### 📈 **Rentabilidad anual - {cartera_sel}**")
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    ax2.bar(anos, r_sel * 100, alpha=0.7, color="darkgreen", edgecolor="black", label="Nominal")
    ax2.plot(anos, r_sel_real * 100, color="red", lw=2, marker="o", label="Real")
    ax2.axhline(0, color="black", lw=1)
    ax2.axhline(cagr(r_sel) * 100, color="blue", ls="--", label=f"CAGR nominal {cagr(r_sel):.1%}")
    ax2.axhline(cagr(r_sel_real) * 100, color="red", ls=":", label=f"CAGR real {cagr(r_sel_real):.1%}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylabel("Rentabilidad (%)")
    pesos_str = {COLS_ACTIVOS[i]: f"{pesos_sel[i]:.0%}" for i in range(len(COLS_ACTIVOS))}
    ax2.set_title(f"Composición {cartera_sel}: {pesos_str}")
    st.pyplot(fig2)

    st.markdown(f"### 🕰️ **Sensibilidad fecha de inicio - {ventana_sens} años**")
    df_sens = sensibilidad_inicio(df_hist, carteras_pesos, ventana_sens)

    if len(df_sens) > 0:
        fig_sens, ax_sens = plt.subplots(figsize=(12, 5))
        for nombre in carteras_pesos.keys():
            sub = df_sens[df_sens["Cartera"] == nombre]
            ax_sens.plot(sub["Inicio"], sub["CAGR"] * 100, lw=2.3, label=nombre)

        ax_sens.set_xlabel("Año de inicio")
        ax_sens.set_ylabel(f"CAGR nominal a {ventana_sens} años (%)")
        ax_sens.set_title(f"Sensibilidad a fecha de inicio comparando carteras ({ventana_sens} años)")
        ax_sens.grid(True, alpha=0.3)
        ax_sens.legend()
        st.pyplot(fig_sens)
    else:
        st.info("No hay suficientes datos para calcular la sensibilidad con esa ventana.")

    st.markdown("### 🌊 **Drawdowns comparativos (underwater)**")
    fig_uw, ax_uw = plt.subplots(figsize=(12, 5))
    for nombre, pesos in carteras_pesos.items():
        r_tmp = rentabilidades_cartera(df_periodo_hist, np.array(pesos))
        _, dd_tmp = serie_drawdown(r_tmp)
        ax_uw.plot(anos, dd_tmp * 100, lw=2, label=nombre)

    ax_uw.axhline(0, color="black", lw=1)
    ax_uw.set_xlabel("Año")
    ax_uw.set_ylabel("Drawdown (%)")
    ax_uw.set_title("Curvas underwater comparativas")
    ax_uw.grid(True, alpha=0.3)
    ax_uw.legend()
    st.pyplot(fig_uw)

    st.markdown("### 🫧 **Scatter drawdown vs recuperación (comparativo)**")
    df_scatter = scatter_drawdown_recovery(df_periodo_hist, carteras_pesos)

    if len(df_scatter) > 0:
        fig_sc, ax_sc = plt.subplots(figsize=(12, 5))
        cmap = plt.get_cmap("tab10")
        nombres = list(carteras_pesos.keys())

        for idx, nombre in enumerate(nombres):
            sub = df_scatter[df_scatter["Cartera"] == nombre]
            if len(sub) > 0:
                ax_sc.scatter(
                    sub["Drawdown"] * 100,
                    sub["Recuperación (años)"],
                    s=90,
                    alpha=0.75,
                    color=cmap(idx),
                    label=nombre,
                    edgecolors="black"
                )

        ax_sc.invert_xaxis()
        ax_sc.set_xlabel("Profundidad del drawdown (%)")
        ax_sc.set_ylabel("Recuperación (años)")
        ax_sc.set_title("Episodios históricos: drawdown vs recuperación")
        ax_sc.grid(True, alpha=0.3)
        ax_sc.legend()
        st.pyplot(fig_sc)
    else:
        st.info("No hay episodios suficientes para el scatter comparativo.")

    st.markdown(f"### 🔥 **Heatmap CAGR - {cartera_sel}**")
    # Recorte a máximo 40 años para que no sea demasiado pesado
    max_anos_heat = 40
    if len(df_periodo_hist) > max_anos_heat:
        df_periodo_heat = df_periodo_hist.iloc[-max_anos_heat:]
    else:
        df_periodo_heat = df_periodo_hist

    anos_heat, mat_heat = heatmap_cagr(df_periodo_heat, pesos_sel)

    fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
    im = ax_hm.imshow(mat_heat * 100, cmap="RdYlGn", aspect="auto", origin="upper")

    ax_hm.set_xticks(np.arange(len(anos_heat)))
    ax_hm.set_yticks(np.arange(len(anos_heat)))
    ax_hm.set_xticklabels(anos_heat, rotation=90, fontsize=8)
    ax_hm.set_yticklabels(anos_heat, fontsize=8)

    ax_hm.set_xlabel("Año final")
    ax_hm.set_ylabel("Año inicial")
    ax_hm.set_title(f"Heatmap de CAGR nominal - {cartera_sel} (periodo seleccionado)")

    cbar = fig_hm.colorbar(im, ax=ax_hm)
    cbar.set_label("CAGR (%)")

    st.pyplot(fig_hm)

# ========================================
# CALCULADORA INTERÉS COMPUESTO
# ========================================
st.markdown("---")
st.header("💶 Calculadora de interés compuesto")

col_ic1, col_ic2, col_ic3, col_ic4 = st.columns(4)

with col_ic1:
    ic_inicial = st.number_input("Capital inicial (€)", 0, 500000, 10000, 1000, key="ic_inicial")
with col_ic2:
    ic_mensual = st.number_input("Aportación mensual (€)", 0, 10000, 500, 50, key="ic_mensual")
with col_ic3:
    ic_anos = st.slider("Años", 1, 50, 30, key="ic_anos")
with col_ic4:
    ic_interes = st.slider("Interés anual (%)", 0.0, 15.0, 7.0, 0.1, key="ic_interes")

df_ic = simular_interes_compuesto(
    aporte_inicial=ic_inicial,
    aporte_mensual=ic_mensual,
    anos=ic_anos,
    interes_anual=ic_interes / 100
)

df_hitos = tabla_hitos_compuesto(df_ic)

saldo_final = df_ic["Saldo"].iloc[-1]
intereses_final = df_ic["Intereses"].iloc[-1]
aportaciones_totales = df_ic["Inicial"].iloc[-1] + df_ic["Aportaciones"].iloc[-1]

colm1, colm2, colm3 = st.columns(3)
with colm1:
    st.metric("Valor final", f"€{saldo_final:,.0f}")
with colm2:
    st.metric("Total aportado", f"€{aportaciones_totales:,.0f}")
with colm3:
    st.metric("Intereses generados", f"€{intereses_final:,.0f}")

st.markdown("### 📈 **Evolución hacia 1 millón**")

fig_ic, ax_ic = plt.subplots(figsize=(12, 6))

ax_ic.stackplot(
    df_ic["Años"],
    df_ic["Inicial"],
    df_ic["Aportaciones"],
    df_ic["Intereses"],
    labels=["Capital inicial", "Aportaciones mensuales", "Intereses"],
    colors=["#4C78A8", "#72B7B2", "#F58518"],
    alpha=0.9
)

for nivel in range(100000, 1000001, 100000):
    ax_ic.axhline(nivel, color="gray", lw=0.8, ls="--", alpha=0.35)

for _, row in df_hitos.dropna(subset=["Años"]).iterrows():
    ax_ic.scatter(row["Años"], row["Hito"], color="black", s=25, zorder=5)
    ax_ic.text(row["Años"], row["Hito"] + 15000, f'{int(row["Hito"]/1000)}k',
               fontsize=8, ha="center")

ax_ic.set_xlabel("Años")
ax_ic.set_ylabel("Patrimonio (€)")
ax_ic.set_title("Crecimiento del patrimonio: inicial + aportaciones + intereses")
ax_ic.grid(True, alpha=0.25)
ax_ic.legend()
st.pyplot(fig_ic)

st.markdown("### 🧮 **Hitos de crecimiento**")

df_hitos_show = df_hitos.copy()
df_hitos_show["Hito"] = df_hitos_show["Hito"].apply(lambda x: f"€{x:,.0f}")
df_hitos_show["Inicial"] = df_hitos_show["Inicial"].apply(lambda x: f"€{x:,.0f}" if pd.notnull(x) else "")
df_hitos_show["Aportaciones"] = df_hitos_show["Aportaciones"].apply(lambda x: f"€{x:,.0f}" if pd.notnull(x) else "")
df_hitos_show["Intereses"] = df_hitos_show["Intereses"].apply(lambda x: f"€{x:,.0f}" if pd.notnull(x) else "")
df_hitos_show["Años"] = df_hitos_show["Años"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
df_hitos_show["Meses"] = df_hitos_show["Meses"].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")

st.dataframe(
    df_hitos_show[[
        "Hito",
        "Tiempo",
        "Desde hito anterior",
        "Años",
        "Meses",
        "Inicial",
        "Aportaciones",
        "Intereses"
    ]],
    use_container_width=True,
    height=390
)

st.markdown("---")
st.caption("🎮 **v2.8.1** | Simulador carteras + interés compuesto | Hitos hasta 1M | Tiempo entre hitos | Desglose de aportaciones e intereses")
