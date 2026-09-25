import os
import sys
import tempfile
import io
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Adicionar o diretório atual ao path para garantir importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from training_runner import (
        train_single_core_generator,
        train_multi_core_generator,
        run_testing_simulation
    )
except ImportError:
    from rl_buyer_interface.training_runner import (
        train_single_core_generator,
        train_multi_core_generator,
        run_testing_simulation
    )

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="RL Buyer Agent - Eureka Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IDIOMA / LANGUAGE SELECTOR ---
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Português"], index=0)

def T(pt, en):
    return en if lang == "English" else pt

# --- CUSTOM CSS: PREMIUM LIGHT ORANGE THEME ---
st.markdown("""
<style>
    /* Configuração Geral de Fontes - Neo Tech Std Medium */
    @font-face {
        font-family: 'Neo Tech Std Medium';
        font-style: normal;
        font-weight: 500;
        src: local('Neo Tech Std Medium'), local('NeoTechStd-Medium'), local('Neo Tech Std');
    }
    
    html, body, [class*="css"], .stApp {
        font-family: 'Neo Tech Std Medium', sans-serif !important;
    }
    
    /* Tema Claro Branco e Laranja */
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }
    
    /* Sidebar styling - Light gray background with orange touch */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom Card container - Clean light border and white background */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Header Gradient styling - Orange and Warm yellow/Darker Orange gradient */
    .main-title {
        background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Terminal Console style - Light terminal or clean dark console */
    .console-header {
        background-color: #0f172a;
        border-radius: 8px 8px 0 0;
        border: 1px solid #1e293b;
        border-bottom: none;
        padding: 8px 15px;
        font-size: 0.85rem;
        color: #ff8c00;
        font-family: monospace;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .console-box {
        background-color: #020617;
        border: 1px solid #1e293b;
        border-radius: 0 0 8px 8px;
        padding: 15px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        color: #38bdf8;
        height: 350px;
        overflow-y: auto;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #64748b;
        border-bottom-width: 2px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ff8c00;
    }
    .stTabs [aria-selected="true"] {
        color: #ff8c00 !important;
        border-bottom-color: #ff8c00 !important;
    }
    
    /* Metric Card Custom styling */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ea580c;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Specialized Spoilage Cards */
    .spoilage-hero-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1.5px solid #86efac;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.12);
        transition: transform 0.2s ease;
    }
    .spoilage-hero-card:hover {
        transform: translateY(-2px);
    }
    .spoilage-standard-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease;
    }
    .spoilage-standard-card:hover {
        transform: translateY(-2px);
    }
    .spoilage-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .badge-green {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .badge-gray {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS FOR FILE ZIP & DOWNLOADS ---
def create_model_zip_bytes(model_base_path):
    """ Lê os arquivos gerados do modelo e empacota-os num buffer ZIP em memória """
    zip_buffer = io.BytesIO()
    suffixes = ['_actor.pth', '_critic.pth', '_scaler.pth', '_econ_stat.pth']
    found_any = False
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for suffix in suffixes:
            file_path = model_base_path + suffix
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))
                found_any = True
                
    if not found_any:
        return None
    return zip_buffer.getvalue()

# --- APP LAYOUT ---

# Top Banner Header
col_logo, col_desc = st.columns([1, 12])
with col_desc:
    st.markdown('<h1 class="main-title">RL Buyer Agent - Eureka</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{T("Interface Interativa de Inteligência Artificial para Gestão Automática de Inventário Fruta & Logística", "Interactive Artificial Intelligence Interface for Automatic Fruit Inventory & Logistics Management")}</p>', unsafe_allow_html=True)

# ----------------- SIDEBAR: DATA UPLOAD & GLOBAL CONFIG -----------------
with st.sidebar:
    st.markdown(f"### {T('📥 Carregar Dataset de Demanda', '📥 Load Demand Dataset')}")
    uploaded_file = st.file_uploader(
        T("Selecione o arquivo Excel ou CSV com dados de vendas históricas e meteorologia:", "Select Excel or CSV file with historical sales and weather data:"),
        type=["xlsx", "csv"],
        help=T("O arquivo deve conter as colunas: real_value, prediction, price, temperature, humidity, ethylene.", "The file must contain the columns: real_value, prediction, price, temperature, humidity, ethylene.")
    )
    
    st.markdown("---")
    st.markdown(f"### ⚙️ {T('Configuração Global', 'Global Configuration')}")
    device_opt = st.selectbox(T("Hardware de Execução (PyTorch):", "Execution Hardware (PyTorch):"), ["CPU", "GPU"], index=0)
    device = "cuda" if device_opt == "GPU" and torch.cuda.is_available() else "cpu"
    if device_opt == "GPU" and not torch.cuda.is_available():
        st.warning(T("Aceleração GPU (CUDA) indisponível. Usando CPU.", "GPU acceleration (CUDA) unavailable. Using CPU."))
        
    st.markdown("---")
    st.markdown(f"#### ℹ️ {T('Estrutura do Dataset Esperada', 'Expected Dataset Structure')}")
    st.info(
        T("**Variáveis Necessárias:**\n"
          "- `real_value`: Vendas Reais\n"
          "- `prediction`: Procura Prevista\n"
          "- `price`: Preço Unitário\n"
          "- `temperature`: Temperatura do dia\n"
          "- `humidity`: Humidade Relativa\n"
          "- `ethylene`: Concentração Etileno",
          "**Required Variables:**\n"
          "- `real_value`: Real Sales\n"
          "- `prediction`: Predicted Demand\n"
          "- `price`: Unit Price\n"
          "- `temperature`: Daily Temperature\n"
          "- `humidity`: Relative Humidity\n"
          "- `ethylene`: Ethylene Concentration")
    )

# ----------------- SESSION STATE INITS -----------------
if 'train_log' not in st.session_state:
    st.session_state.train_log = ""
if 'test_log' not in st.session_state:
    st.session_state.test_log = ""
if 'trained_model_dir' not in st.session_state:
    st.session_state.trained_model_dir = None
if 'test_completed' not in st.session_state:
    st.session_state.test_completed = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'holding_cost' not in st.session_state:
    st.session_state.holding_cost = 0.70
if 'transport_cost' not in st.session_state:
    st.session_state.transport_cost = 10.00
if 'fixed_transport_cost' not in st.session_state:
    st.session_state.fixed_transport_cost = 10.00
if 'stockout_penalty_pct' not in st.session_state:
    st.session_state.stockout_penalty_pct = 25.0
if 'waste_penalty_pct' not in st.session_state:
    st.session_state.waste_penalty_pct = 100.0
if 'zero_stock_penalty_pct' not in st.session_state:
    st.session_state.zero_stock_penalty_pct = 500.0

# Verificação inicial se há dataset carregado
if uploaded_file is None:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.warning(T("👋 Por favor, faça upload de um ficheiro de dataset na barra lateral para começar a interagir com o agente de compras RL.", "👋 Please upload a dataset file in the sidebar to start interacting with the RL buyer agent."))
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Carregar o arquivo na memória
@st.cache_data
def load_uploaded_df(file_obj):
    if file_obj.name.endswith('.xlsx'):
        return pd.read_excel(file_obj)
    else:
        return pd.read_csv(file_obj)

df_data = load_uploaded_df(uploaded_file)
st.sidebar.success(T(f"Dataset carregado com {len(df_data)} dias de registos!", f"Dataset loaded with {len(df_data)} days of records!"))

# Criar arquivo temporário local para a leitura das rotinas do modelo
temp_dataset_path = os.path.join(tempfile.gettempdir(), "uploaded_dataset.xlsx")
df_data.to_excel(temp_dataset_path, index=False)

# ----------------- TABS PRINCIPAIS -----------------
tab_train, tab_test = st.tabs([T("🏋️ Treinar Modelo", "🏋️ Train Model"), T("🧪 Simulação & Inferência", "🧪 Simulation & Inference")])

# =====================================================================
# TAB 1: TREINO DO MODELO
# =====================================================================
with tab_train:
    st.markdown(f"### {T('Treino Offline do Modelo PPO', 'Offline Training of the PPO Model')}")
    st.markdown(T("Configure as variáveis de ambiente e os hiperparâmetros mais importantes do algoritmo Proximal Policy Optimization (PPO) abaixo.", "Configure environmental variables and the most important hyperparameters of the Proximal Policy Optimization (PPO) algorithm below."))
    
    col_split, col_cap = st.columns(2)
    with col_split:
        train_split = st.slider(T("Percentagem (%) de Dados para Treino:", "Percentage (%) of Data for Training:"), min_value=30, max_value=90, value=60, step=5,
                                help=T("Apenas esta fração inicial do dataset será usada para treinar o agente. O restante é reservado para teste.", "Only this initial fraction of the dataset will be used to train the agent. The rest is reserved for testing."))
    with col_cap:
        max_capacity = st.number_input(T("Capacidade Máxima do Armazém (unidades):", "Maximum Warehouse Capacity (units):"), min_value=100, max_value=5000, value=500, step=100)
        
    # Hiperparâmetros de RL pré-definidos (escondidos do utilizador para simplificar a UI de negócios)
    lr_act = 0.0003
    lr_crit = 0.001
    gamma = 0.8
    eps_clip = 0.2
    k_epochs = 30
    batch_size = 2048
    max_episodes = 1000
    seed = 1337

    st.markdown(f"#### 🎯 {T('Função Objetivo (Lucro Diário)', 'Objective Function (Daily Profit)')}")
    st.markdown(T(
        "A recompensa diária do agente RL é baseada no lucro líquido otimizado pela seguinte função objetivo:",
        "The daily reward of the RL agent is based on the net profit optimized by the following objective function:"
    ))
    
    if lang == "English":
        st.latex(r"""
        \text{Profit} = (\text{Sales} \cdot P) - \text{Costs}_{\text{storage}} - \text{Costs}_{\text{trans}} - \text{Penalty}_{\text{lost sales}} - \text{Penalty}_{\text{waste}} - \text{Penalty}_{\text{stockout}}
        """)
    else:
        st.latex(r"""
        \text{Lucro} = (\text{Vendas} \cdot P) - \text{Custos}_{\text{arm}} - \text{Custos}_{\text{trans}} - \text{Penalização}_{\text{vendas perdidas}} - \text{Penalização}_{\text{desperdício}} - \text{Penalização}_{\text{stockout}}
        """)
    
    with st.expander(T("🔎 Detalhes Matemáticos da Equação", "🔎 Mathematical Equation Details")):
        if lang == "English":
            st.latex(r"""
            \begin{aligned}
            \text{Costs}_{\text{storage}} &= \text{Final Stock} \cdot V_{\text{prod}} \cdot C_{\text{storage}} \\
            \text{Costs}_{\text{trans}} &= \mathbb{I}_{\{\text{Order} > 0\}} \cdot \left( F_{\text{trans}} + \text{Order} \cdot V_{\text{prod}} \cdot C_{\text{trans}} \right) \\
            \text{Penalty}_{\text{lost sales}} &= \text{Lost Sales} \cdot P \cdot \alpha \\
            \text{Penalty}_{\text{waste}} &= (\text{Spoilage} + \text{Overflow}) \cdot P \cdot \beta \\
            \text{Penalty}_{\text{stockout}} &= \mathbb{I}_{\{\text{Final Stock} \le 0\}} \cdot P \cdot \gamma
            \end{aligned}
            """)
        else:
            st.latex(r"""
            \begin{aligned}
            \text{Custos}_{\text{arm}} &= \text{Stock Final} \cdot V_{\text{prod}} \cdot C_{\text{arm}} \\
            \text{Custos}_{\text{trans}} &= \mathbb{I}_{\{\text{Encomenda} > 0\}} \cdot \left( F_{\text{trans}} + \text{Encomenda} \cdot V_{\text{prod}} \cdot C_{\text{trans}} \right) \\
            \text{Penalização}_{\text{vendas perdidas}} &= \text{Procura Perdida} \cdot P \cdot \alpha \\
            \text{Penalização}_{\text{desperdício}} &= (\text{Apodrecimento} + \text{Excesso}) \cdot P \cdot \beta \\
            \text{Penalização}_{\text{stockout}} &= \mathbb{I}_{\{\text{Stock Final} \le 0\}} \cdot P \cdot \gamma
            \end{aligned}
            """)
        st.markdown(T(
            r"onde $P$ é o preço unitário do SKU, $V_{\text{prod}}$ é o volume do produto ($m^3$), "
            r"$C_{\text{arm}}$ é o Custo de Armazenamento, $C_{\text{trans}}$ é o Custo de Transporte, "
            r"$F_{\text{trans}}$ é a Taxa Fixa, $\alpha$ é a Penalização de Vendas Perdidas, "
            r"$\beta$ é a Penalização de Desperdício e $\gamma$ é a Penalização de Stockout.",
            r"where $P$ is the SKU unit price, $V_{\text{prod}}$ is the product volume ($m^3$), "
            r"$C_{\text{arm}}$ is the Storage Cost, $C_{\text{trans}}$ is the Transport Cost, "
            r"$F_{\text{trans}}$ is the Fixed Fee, $\alpha$ is the Lost Sales Penalty, "
            r"$\beta$ is the Waste Penalty, and $\gamma$ is the Stockout Penalty."
        ))

    st.markdown(f"#### ⚖️ {T('Custos de Logística e Penalizações', 'Logistics Costs & Penalties')}")
    col_hp1, col_hp2, col_hp3 = st.columns(3)
    with col_hp1:
        holding_cost = st.number_input(T("Custo Armazenamento (€/m³/dia):", "Storage Cost (€/m³/day):"), min_value=0.0, max_value=10.0, value=0.70, step=0.05, format="%.2f", key="holding_cost", help=T("Custo diário de armazenar 1m³ de produto.", "Daily cost of storing 1m³ of product."))
        stockout_penalty_val = st.number_input(T("Penalização Vendas Perdidas (% preço):", "Lost Sales Penalty (% price):"), min_value=0.0, max_value=200.0, value=25.0, step=5.0, key="stockout_penalty_pct") / 100.0
    with col_hp2:
        transport_cost = st.number_input(T("Custo Transporte (€/m³):", "Transport Cost (€/m³):"), min_value=0.0, max_value=100.0, value=10.00, step=0.50, format="%.2f", key="transport_cost", help=T("Custo variável de transportar 1m³ no camião.", "Variable cost of transporting 1m³ in the truck."))
        waste_penalty_val = st.number_input(T("Penalização Desperdício (% preço):", "Waste Penalty (% price):"), min_value=0.0, max_value=500.0, value=100.0, step=10.0, key="waste_penalty_pct") / 100.0
    with col_hp3:
        fixed_transport_cost = st.number_input(T("Taxa Fixa Camião (€):", "Fixed Truck Fee (€):"), min_value=0.0, max_value=500.0, value=10.00, step=1.00, format="%.2f", key="fixed_transport_cost", help=T("Taxa fixa cobrada por descarga/entrega.", "Fixed fee charged per unloading/delivery."))
        zero_stock_penalty_val = st.number_input(T("Penalização Stockout / Stock Zero (% preço):", "Stockout / Zero Stock Penalty (% price):"), min_value=0.0, max_value=1000.0, value=500.0, step=50.0, key="zero_stock_penalty_pct") / 100.0
        
    st.markdown(f"#### {T('Configuração de Hardware & Paralelização', 'Hardware Configuration & Parallelization')}")
    col_hw1, col_hw2 = st.columns(2)
    with col_hw1:
        num_envs = st.slider(T("Número de Ambientes Paralelos:", "Number of Parallel Environments:"), min_value=4, max_value=128, value=64, step=4,
                             help=T("Quantidade de simulações concorrentes para recolha de trajetórias.", "Amount of concurrent simulations for trajectory collection."))
    with col_hw2:
        # Wrap radio options in translated list
        core_mode = st.radio(T("Modo de Processamento:", "Processing Mode:"), [T("Single-Core (Estável em Cloud)", "Single-Core (Stable in Cloud)"), T("Multi-Core (Mais Rápido Localmente)", "Multi-Core (Faster Locally)")], index=0)
        
    workers = 4
    if "Multi-Core" in core_mode or "Faster Locally" in core_mode:
        workers = st.slider(T("Número de Cores Físicos (Workers):", "Number of Physical Cores (Workers):"), min_value=2, max_value=16, value=4, step=2)

    # Iniciar Treino
    btn_train = st.button(T("🚀 Iniciar Treino do Agente", "🚀 Start Agent Training"), use_container_width=True)
    
    # Contentor de logs
    st.markdown(f'<div class="console-header">📁 {T("Terminal do PPO Agent - Logs de Treino", "PPO Agent Terminal - Training Logs")}</div>', unsafe_allow_html=True)
    console_placeholder = st.empty()
    console_placeholder.markdown(f'<div class="console-box">{T("Agente à espera de comando...", "Agent waiting for command...")}</div>', unsafe_allow_html=True)
    
    if btn_train:
        st.session_state.train_log = ""
        temp_model_dir = tempfile.mkdtemp()
        st.session_state.trained_model_dir = temp_model_dir
        
        # Seleciona o gerador de acordo com o modo de cores
        if "Single-Core" in core_mode:
            gen = train_single_core_generator(
                seed=seed,
                excel_path=temp_dataset_path,
                train_split=(train_split / 100.0),
                max_capacity=max_capacity,
                lr_actor=lr_act,
                lr_critic=lr_crit,
                gamma=gamma,
                k_epochs=k_epochs,
                eps_clip=eps_clip,
                batch_size=batch_size,
                max_episodes_total=max_episodes,
                num_envs=num_envs,
                save_dir=temp_model_dir,
                holding_cost=holding_cost,
                transport_cost=transport_cost,
                fixed_transport_cost=fixed_transport_cost,
                stockout_penalty=stockout_penalty_val,
                waste_penalty=waste_penalty_val,
                zero_stock_penalty=zero_stock_penalty_val
            )
        else:
            gen = train_multi_core_generator(
                seed=seed,
                excel_path=temp_dataset_path,
                train_split=(train_split / 100.0),
                max_capacity=max_capacity,
                lr_actor=lr_act,
                lr_critic=lr_crit,
                gamma=gamma,
                k_epochs=k_epochs,
                eps_clip=eps_clip,
                batch_size=batch_size,
                max_episodes_total=max_episodes,
                num_envs=num_envs,
                num_workers=workers,
                save_dir=temp_model_dir,
                holding_cost=holding_cost,
                transport_cost=transport_cost,
                fixed_transport_cost=fixed_transport_cost,
                stockout_penalty=stockout_penalty_val,
                waste_penalty=waste_penalty_val,
                zero_stock_penalty=zero_stock_penalty_val
            )
            
        progress_bar = st.progress(0.0)
        
        # Consome as linhas do gerador em tempo real
        for log_line in gen:
            st.session_state.train_log += log_line + "\n"
            # Formatar e injetar na caixa de terminal
            console_placeholder.markdown(
                f'<div class="console-box">{st.session_state.train_log}</div>',
                unsafe_allow_html=True
            )
            
            # Atualizar barra de progresso simples
            if "Episodes:" in log_line:
                try:
                    parts = log_line.split("Episodes:")[1].split("/")[0].strip()
                    current_ep = int(parts)
                    pct = min(1.0, current_ep / max_episodes)
                    progress_bar.progress(pct)
                except:
                    pass
        
        progress_bar.progress(1.0)
        st.success(T("🎉 Treino concluído com sucesso!", "🎉 Training completed successfully!"))
        
    # Exibir o botão de download caso os arquivos já existam
    if st.session_state.trained_model_dir is not None:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"#### 💾 {T('Guardar Ficheiros Finais do Modelo', 'Save Final Model Files')}")
        st.write(T("Transfira o modelo treinado com os pesos do Actor, Critic, Scaler e estatísticas Z-Score compilados num ficheiro ZIP:", "Download the trained model with Actor, Critic, Scaler weights and compiled Z-Score statistics in a ZIP file:"))
        
        # Gerar o arquivo ZIP em memória
        base_path = os.path.join(st.session_state.trained_model_dir, "ppo_constrained_final")
        zip_bytes = create_model_zip_bytes(base_path)
        
        if zip_bytes is not None:
            st.download_button(
                label=T("📥 Descarregar Ficheiros de Treino (.ZIP)", "📥 Download Training Files (.ZIP)"),
                data=zip_bytes,
                file_name="modelo_ppo_buyer_agente.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            # Exibir gráfico de losses se existir
            plot_path = os.path.join(st.session_state.trained_model_dir, "losses_plot.png")
            if os.path.exists(plot_path):
                st.markdown("---")
                st.markdown(f"##### {T('Curva de Aprendizagem (Loss Evolution)', 'Learning Curve (Loss Evolution)')}")
                st.image(plot_path, caption=T("Gráficos de Losses de Treino (Actor, Critic e Total)", "Training Loss Charts (Actor, Critic and Total)"), use_container_width=True)
        else:
            st.error(T("Não foram encontrados ficheiros de pesos do modelo no diretório temporário.", "No model weight files found in the temporary directory."))
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================
# TAB 2: TESTE E FINE-TUNING DO MODELO
# =====================================================================
with tab_test:
    st.markdown(f"### {T('Simulação em Produção & Fine-Tuning Contínuo', 'Production Simulation & Continuous Fine-Tuning')}")
    st.markdown(T("Execute a simulação no período de teste do dataset e avalie o desempenho do RL Agent em tempo real face a estratégias Baseline.", "Run the simulation on the test period of the dataset and evaluate the RL Agent's performance in real time against Baseline strategies."))
    
    col_ds, col_model = st.columns(2)
    with col_ds:
        st.markdown('<div class="custom-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown(f"##### {T('1. Configurar Dados de Teste', '1. Configure Test Data')}")
        test_split_option = st.radio(
            T("Origem dos dias de teste:", "Source of test days:"),
            [T("Usar os restantes (100 - X)% do dataset de treino carregado", "Use the remaining (100 - X)% of the loaded training dataset"), T("Carregar outro ficheiro com novos dias", "Load another file with new days")]
        )
        
        test_split_val = 0.6
        if "Usar os restantes" in test_split_option or "Use the remaining" in test_split_option:
            test_split_val = train_split / 100.0
            st.info(T(f"A simulação usará a fração restante dos dados ({100 - train_split}% de {len(df_data)} dias = {int(len(df_data) * (1 - test_split_val))} dias).", f"The simulation will use the remaining data fraction ({100 - train_split}% of {len(df_data)} days = {int(len(df_data) * (1 - test_split_val))} days)."))
            final_test_dataset_path = temp_dataset_path
        else:
            uploaded_test_file = st.file_uploader(
                T("Carregar Dataset de Teste / Futuro:", "Upload Test / Future Dataset:"),
                type=["xlsx", "csv"],
                key="test_dataset_uploader"
            )
            if uploaded_test_file is not None:
                df_test = load_uploaded_df(uploaded_test_file)
                st.success(T(f"Novos dias de teste carregados ({len(df_test)} dias).", f"New test days loaded ({len(df_test)} days)."))
                # Escrever ficheiro temporário para teste
                final_test_dataset_path = os.path.join(tempfile.gettempdir(), "test_dataset.xlsx")
                df_test.to_excel(final_test_dataset_path, index=False)
                test_split_val = 0.0 # Todo o ficheiro novo é usado para o teste
            else:
                st.warning(T("A aguardar carregamento do ficheiro de teste...", "Waiting for test file upload..."))
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_model:
        st.markdown('<div class="custom-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown(f"##### {T('2. Upload do Modelo Treinado', '2. Upload Trained Model')}")
        st.write(T("Faça upload do ficheiro ZIP ou dos ficheiros individuais (.pth) contendo os pesos treinados:", "Upload the ZIP file or individual files (.pth) containing the trained weights:"))
        
        upload_mode = st.radio(T("Tipo de upload:", "Upload type:"), [T("Ficheiro ZIP", "ZIP File"), T("Ficheiros Individuais (.pth)", "Individual Files (.pth)")])
        
        temp_load_dir = tempfile.mkdtemp()
        model_loaded_ok = False
        
        if upload_mode == "Ficheiro ZIP" or upload_mode == "ZIP File":
            zip_model_file = st.file_uploader(T("Upload do Modelo ZIP:", "Upload ZIP Model:"), type=["zip"])
            if zip_model_file is not None:
                try:
                    with zipfile.ZipFile(zip_model_file) as zf:
                        zf.extractall(temp_load_dir)
                    # Verificar se encontramos o ficheiro do ator
                    files_in_dir = os.listdir(temp_load_dir)
                    actor_file = [f for f in files_in_dir if f.endswith('_actor.pth')]
                    if len(actor_file) > 0:
                        base_model_name = actor_file[0].split('_actor.pth')[0]
                        initial_model_base_path = os.path.join(temp_load_dir, base_model_name)
                        model_loaded_ok = True
                        st.success(T(f"Modelo ZIP descompactado! Identificado: {base_model_name}", f"ZIP model unpacked! Identified: {base_model_name}"))
                    else:
                        st.error(T("O ficheiro ZIP não contém arquivos no formato esperado (*_actor.pth).", "The ZIP file does not contain files in the expected format (*_actor.pth)."))
                except Exception as e:
                    st.error(T(f"Erro ao descompactar o ZIP: {e}", f"Error unpacking ZIP: {e}"))
        else:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                uploaded_actor = st.file_uploader(T("Ficheiro Actor (*_actor.pth)", "Actor File (*_actor.pth)"), type=["pth"])
                uploaded_scaler = st.file_uploader(T("Ficheiro Scaler (*_scaler.pth) [Opcional]", "Scaler File (*_scaler.pth) [Optional]"), type=["pth"])
            with col_a2:
                uploaded_critic = st.file_uploader(T("Ficheiro Critic (*_critic.pth) [Opcional]", "Critic File (*_critic.pth) [Optional]"), type=["pth"])
                uploaded_econ = st.file_uploader(T("Ficheiro Econ Stat (*_econ_stat.pth)", type=["pth"]))
                
            if uploaded_actor is not None and uploaded_econ is not None:
                # Gravar com sufixos corretos no diretório temporário
                with open(os.path.join(temp_load_dir, "my_model_actor.pth"), "wb") as f:
                    f.write(uploaded_actor.getbuffer())
                with open(os.path.join(temp_load_dir, "my_model_econ_stat.pth"), "wb") as f:
                    f.write(uploaded_econ.getbuffer())
                    
                if uploaded_critic is not None:
                    with open(os.path.join(temp_load_dir, "my_model_critic.pth"), "wb") as f:
                        f.write(uploaded_critic.getbuffer())
                if uploaded_scaler is not None:
                    with open(os.path.join(temp_load_dir, "my_model_scaler.pth"), "wb") as f:
                        f.write(uploaded_scaler.getbuffer())
                        
                initial_model_base_path = os.path.join(temp_load_dir, "my_model")
                model_loaded_ok = True
                st.success(T("Pesos do Actor e do Econ Stat carregados com sucesso!", "Actor and Econ Stat weights loaded successfully!"))
            else:
                st.info(T("Para correr o teste, faça upload do Actor (*_actor.pth) e do Econ Stat (*_econ_stat.pth).", "To run the test, upload the Actor (*_actor.pth) and Econ Stat (*_econ_stat.pth)."))
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown(f"#### {T('Parâmetros de Simulação', 'Simulation Parameters')}")
    col_t1, = st.columns(1)
    with col_t1:
        s_min = st.number_input(T("Valor Mínimo Baseline (s):", "Minimum Baseline Value (s):"), min_value=1, max_value=500, value=24, help=T("Desencadeia encomenda no Baseline Min-Max se o stock estimado for inferior a este valor.", "Triggers an order in the Min-Max Baseline if the estimated stock is lower than this value."))
        S_max = st.number_input(T("Valor Máximo Baseline (S):", "Maximum Baseline Value (S):"), min_value=50, max_value=2000, value=60, help=T("Nível de stock alvo após encomenda no Baseline Min-Max.", "Target stock level after ordering in the Min-Max Baseline."))
        
    # Parâmetros de fine-tuning fixos na retaguarda
    update_interval = 15
    online_batch = 32
    online_lr_act = 1e-5
    online_lr_crit = 5e-5


    # Botão para correr simulação
    st.markdown("---")
    btn_disabled = not model_loaded_ok
    btn_test = st.button(T("🏁 Iniciar Teste e Fine-Tuning Contínuo", "🏁 Start Test & Continuous Fine-Tuning"), use_container_width=True, disabled=btn_disabled)
    
    if btn_disabled:
        st.warning(T("⚠️ O botão de teste está desativado. Por favor, carregue os pesos do modelo (Actor e Econ Stat) primeiro.", "⚠️ The test button is disabled. Please upload the model weights (Actor and Econ Stat) first."))

    # Contentores dinâmicos
    col_chart, col_side_logs = st.columns([8, 4])
    with col_chart:
        chart_placeholder = st.empty()
        chart_placeholder_2 = st.empty()
        chart_placeholder_3 = st.empty()
        
    with col_side_logs:
        st.markdown(f'<div class="console-header">🤖 {T("Logs Diários da Simulação", "Daily Simulation Logs")}</div>', unsafe_allow_html=True)
        test_console_placeholder = st.empty()
        test_console_placeholder.markdown(f'<div class="console-box">{T("Simulação parada...", "Simulation stopped...")}</div>', unsafe_allow_html=True)

    if btn_test and model_loaded_ok:
        st.session_state.test_log = ""
        st.session_state.test_completed = False
        
        # Instanciar o gerador da simulação
        sim_gen = run_testing_simulation(
            excel_path=final_test_dataset_path,
            train_split=test_split_val,
            max_capacity=max_capacity,
            initial_model_base_path=initial_model_base_path,
            s_min=s_min,
            S_max=S_max,
            update_interval_days=update_interval,
            online_lr_actor=online_lr_act,
            online_lr_critic=online_lr_crit,
            online_batch_size=online_batch,
            save_dir=temp_load_dir,
            holding_cost=st.session_state.holding_cost,
            transport_cost=st.session_state.transport_cost,
            fixed_transport_cost=st.session_state.fixed_transport_cost,
            stockout_penalty=st.session_state.stockout_penalty_pct / 100.0,
            waste_penalty=st.session_state.waste_penalty_pct / 100.0,
            zero_stock_penalty=st.session_state.zero_stock_penalty_pct / 100.0
        )
        
        # Preparar dataframes para a atualização gráfica em tempo real
        plot_data = {
            "Dia": [],
            "RL Agent": [],
            "Min-Max": [],
            "Oracle": [],
            "Real Demand": [],
            "Stock Level": [],
            "Orders": [],
            "Spoilage": [],
            "MinMax Stock Level": [],
            "MinMax Orders": [],
            "MinMax Spoilage": []
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=T('RL Agent (Profit Acumulado)', 'RL Agent (Cumulative Profit)'), line=dict(color='#10b981', width=2.5)))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=T('Min-Max Baseline', 'Min-Max Baseline'), line=dict(color='#8c9cb3', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=T('Oráculo (God Mode)', 'Oracle (God Mode)'), line=dict(color='#ffaa00', width=2.0, dash='dash')))
        fig.update_layout(
            title=T("Evolução Comparativa do Lucro Acumulado em Tempo Real", "Comparative Real-Time Cumulative Profit Evolution"),
            xaxis_title=T("Dias", "Days"),
            yaxis_title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1e293b"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=[], y=[], name=T('Encomendas do Agente', 'Agent Orders'), marker=dict(color='rgba(245, 158, 11, 0.75)', line=dict(color='#d97706', width=1))))
        fig2.add_trace(go.Scatter(x=[], y=[], mode='lines', name=T('Procura Real', 'Real Demand'), line=dict(color='#3b82f6', width=2)))
        fig2.add_trace(go.Scatter(x=[], y=[], mode='markers', name=T('Stockout', 'Stockout'), marker=dict(symbol='circle', color='#ffffff', size=8, line=dict(color='#000000', width=1.5))))
        fig2.add_trace(go.Scatter(x=[], y=[], mode='markers', name=T('Apodrecimento', 'Spoilage'), marker=dict(symbol='diamond', color='#ef4444', size=8, line=dict(color='#991b1b', width=1.5))))
        fig2.update_layout(
            title=T("Fluxo Operacional Diário - RL Agent (Procura, Encomendas e Penalidades)", "Daily Operational Flow - RL Agent (Demand, Orders & Penalties)"),
            xaxis_title=T("Dias", "Days"),
            yaxis_title=T("Quantidade (unidades)", "Quantity (units)"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1e293b"),
            barmode='group',
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        chart_placeholder_2.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=[], y=[], name=T('Encomendas Min-Max', 'Min-Max Orders'), marker=dict(color='rgba(140, 156, 179, 0.75)', line=dict(color='#64748b', width=1))))
        fig3.add_trace(go.Scatter(x=[], y=[], mode='lines', name=T('Procura Real', 'Real Demand'), line=dict(color='#3b82f6', width=2)))
        fig3.add_trace(go.Scatter(x=[], y=[], mode='markers', name=T('Stockout', 'Stockout'), marker=dict(symbol='circle', color='#ffffff', size=8, line=dict(color='#000000', width=1.5))))
        fig3.add_trace(go.Scatter(x=[], y=[], mode='markers', name=T('Apodrecimento', 'Spoilage'), marker=dict(symbol='diamond', color='#ef4444', size=8, line=dict(color='#991b1b', width=1.5))))
        fig3.update_layout(
            title=T("Fluxo Operacional Diário - Min-Max (Procura, Encomendas e Penalidades)", "Daily Operational Flow - Min-Max (Demand, Orders & Penalties)"),
            xaxis_title=T("Dias", "Days"),
            yaxis_title=T("Quantidade (unidades)", "Quantity (units)"),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1e293b"),
            barmode='group',
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        chart_placeholder_3.plotly_chart(fig3, use_container_width=True)
        
        for sim_step in sim_gen:
            status = sim_step.get("status")
            msg = sim_step.get("msg")
            
            if status in ["init", "start", "error", "warning"]:
                st.session_state.test_log += msg + "\n"
                test_console_placeholder.markdown(
                    f'<div class="console-box">{st.session_state.test_log}</div>',
                    unsafe_allow_html=True
                )
                if status == "error":
                    st.error(T("Erro crítico na inicialização.", "Critical initialization error."))
                    break
                    
            elif status == "running":
                # Print log to console
                st.session_state.test_log += msg + "\n"
                # Keep scroll down locally:
                # split and keep last 20 lines to avoid slow page updates
                lines = st.session_state.test_log.split("\n")
                visible_logs = "\n".join(lines[-25:])
                test_console_placeholder.markdown(
                    f'<div class="console-box">{visible_logs}</div>',
                    unsafe_allow_html=True
                )
                
                # Atualizar dados gráficos
                plot_data["Dia"].append(sim_step["day"])
                plot_data["RL Agent"].append(sim_step["agent_profit_cum"])
                plot_data["Min-Max"].append(sim_step["minmax_profit_cum"])
                plot_data["Oracle"].append(sim_step["oracle_profit_cum"])
                plot_data["Real Demand"].append(sim_step["real_demand"])
                plot_data["Stock Level"].append(sim_step["stock_level"])
                plot_data["Orders"].append(sim_step["order_placed"])
                plot_data["Spoilage"].append(sim_step["spoilage"])
                plot_data["MinMax Stock Level"].append(sim_step["minmax_stock_level"])
                plot_data["MinMax Orders"].append(sim_step["minmax_action"])
                plot_data["MinMax Spoilage"].append(sim_step["minmax_spoilage"])
                
                # Fazer o redesenho parcial a cada 3 dias para suavizar performance do Streamlit
                if sim_step["day"] % 3 == 0 or sim_step["update_triggered"]:
                    # Criar nova figure com os arrays acumulados
                    fig_real = go.Figure()
                    fig_real.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["RL Agent"], mode='lines', name=f'RL Agent ({plot_data["RL Agent"][-1]:.0f}€)', line=dict(color='#10b981', width=2.5)))
                    fig_real.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Min-Max"], mode='lines', name=f'Min-Max ({plot_data["Min-Max"][-1]:.0f}€)', line=dict(color='#8c9cb3', width=1.5, dash='dot')))
                    fig_real.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Oracle"], mode='lines', name=f'Oráculo ({plot_data["Oracle"][-1]:.0f}€)', line=dict(color='#ffaa00', width=2.0, dash='dash')))
                    
                    fig_real.update_layout(
                        title=T("Evolução Comparativa do Lucro Acumulado em Tempo Real", "Comparative Real-Time Cumulative Profit Evolution"),
                        xaxis_title=T("Dias", "Days"),
                        yaxis_title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        font=dict(color="#1e293b"),
                        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                        margin=dict(t=50, b=40, l=40, r=40)
                    )
                    chart_placeholder.plotly_chart(fig_real, use_container_width=True)

                    # Calcular pontos de penalidades para o gráfico operacional do Agente
                    stockout_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["Stock Level"][idx] <= 0]
                    stockout_y = [plot_data["Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["Stock Level"][idx] <= 0]
                    
                    spoilage_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["Spoilage"][idx] > 0]
                    spoilage_y = [plot_data["Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["Spoilage"][idx] > 0]

                    # Criar novo gráfico operacional
                    fig_ops = go.Figure()
                    fig_ops.add_trace(go.Bar(x=plot_data["Dia"], y=plot_data["Orders"], name=T('Encomendas do Agente', 'Agent Orders'), marker=dict(color='rgba(245, 158, 11, 0.75)', line=dict(color='#d97706', width=1))))
                    fig_ops.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Real Demand"], mode='lines', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)', name=T('Procura Real', 'Real Demand'), line=dict(color='#3b82f6', width=2, shape='spline')))
                    fig_ops.add_trace(go.Scatter(x=stockout_x, y=stockout_y, mode='markers', name=T('Stockout', 'Stockout'), marker=dict(symbol='circle', color='#ffffff', size=8, line=dict(color='#000000', width=1.5))))
                    fig_ops.add_trace(go.Scatter(x=spoilage_x, y=spoilage_y, mode='markers', name=T('Apodrecimento', 'Spoilage'), marker=dict(symbol='diamond', color='#ef4444', size=8, line=dict(color='#991b1b', width=1.5))))
                    
                    fig_ops.update_layout(
                        title=T("Fluxo Operacional Diário - RL Agent (Procura, Encomendas e Penalidades)", "Daily Operational Flow - RL Agent (Demand, Orders & Penalties)"),
                        xaxis_title=T("Dias", "Days"),
                        yaxis_title=T("Quantidade (unidades)", "Quantity (units)"),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        font=dict(color="#1e293b"),
                        barmode='group',
                        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                        margin=dict(t=50, b=40, l=40, r=40)
                    )
                    chart_placeholder_2.plotly_chart(fig_ops, use_container_width=True)

                    # Calcular pontos de penalidades para o gráfico operacional do Min-Max
                    mm_stockout_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Stock Level"][idx] <= 0]
                    mm_stockout_y = [plot_data["MinMax Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Stock Level"][idx] <= 0]
                    
                    mm_spoilage_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Spoilage"][idx] > 0]
                    mm_spoilage_y = [plot_data["MinMax Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Spoilage"][idx] > 0]

                    # Criar novo gráfico operacional Min-Max
                    fig_ops_mm = go.Figure()
                    fig_ops_mm.add_trace(go.Bar(x=plot_data["Dia"], y=plot_data["MinMax Orders"], name=T('Encomendas Min-Max', 'Min-Max Orders'), marker=dict(color='rgba(140, 156, 179, 0.75)', line=dict(color='#64748b', width=1))))
                    fig_ops_mm.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Real Demand"], mode='lines', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)', name=T('Procura Real', 'Real Demand'), line=dict(color='#3b82f6', width=2, shape='spline')))
                    fig_ops_mm.add_trace(go.Scatter(x=mm_stockout_x, y=mm_stockout_y, mode='markers', name=T('Stockout', 'Stockout'), marker=dict(symbol='circle', color='#ffffff', size=8, line=dict(color='#000000', width=1.5))))
                    fig_ops_mm.add_trace(go.Scatter(x=mm_spoilage_x, y=mm_spoilage_y, mode='markers', name=T('Apodrecimento', 'Spoilage'), marker=dict(symbol='diamond', color='#ef4444', size=8, line=dict(color='#991b1b', width=1.5))))
                    
                    fig_ops_mm.update_layout(
                        title=T("Fluxo Operacional Diário - Min-Max (Procura, Encomendas e Penalidades)", "Daily Operational Flow - Min-Max (Demand, Orders & Penalties)"),
                        xaxis_title=T("Dias", "Days"),
                        yaxis_title=T("Quantidade (unidades)", "Quantity (units)"),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        font=dict(color="#1e293b"),
                        barmode='group',
                        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                        margin=dict(t=50, b=40, l=40, r=40)
                    )
                    chart_placeholder_3.plotly_chart(fig_ops_mm, use_container_width=True)
                    
            elif status == "complete":
                st.session_state.test_log += msg + "\n"
                test_console_placeholder.markdown(
                    f'<div class="console-box">{st.session_state.test_log}</div>',
                    unsafe_allow_html=True
                )
                
                # Guardar resultados no session state
                st.session_state.test_completed = True
                st.session_state.test_results = sim_step
                st.session_state.test_plot_data = plot_data
                
                # Plot final
                fig_final = go.Figure()
                fig_final.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["RL Agent"], mode='lines', name=f'RL Agent ({sim_step["cum_profit_agent"]:.1f}€)', line=dict(color='#10b981', width=3.0)))
                fig_final.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Min-Max"], mode='lines', name=f'Min-Max ({sim_step["cum_profit_minmax"]:.1f}€)', line=dict(color='#8c9cb3', width=1.5, dash='dot')))
                fig_final.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Oracle"], mode='lines', name=f'Oráculo ({sim_step["cum_profit_oracle"]:.1f}€)', line=dict(color='#ffaa00', width=2.0, dash='dash')))
                
                # Marcar linhas de fine-tuning
                for ud in sim_step["update_days"]:
                    fig_final.add_vline(x=ud, line_width=1, line_dash="dash", line_color="#ff4757", annotation_text="Fine-Tuning", annotation_position="top left")
                
                fig_final.update_layout(
                    title=T("Evolução Comparativa do Lucro Acumulado Final", "Comparative Final Cumulative Profit Evolution"),
                    xaxis_title=T("Dias", "Days"),
                    yaxis_title=T("Lucro Acumulado (€)", "Cumulative Profit (€)"),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#1e293b"),
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                )
                chart_placeholder.plotly_chart(fig_final, use_container_width=True)

                # Plot final operacional
                # Calcular pontos de penalidades para o gráfico operacional
                stockout_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["Stock Level"][idx] <= 0]
                stockout_y = [plot_data["Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["Stock Level"][idx] <= 0]
                
                spoilage_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["Spoilage"][idx] > 0]
                spoilage_y = [plot_data["Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["Spoilage"][idx] > 0]

                fig_final_ops = go.Figure()
                fig_final_ops.add_trace(go.Bar(x=plot_data["Dia"], y=plot_data["Orders"], name=T('Encomendas do Agente', 'Agent Orders'), marker=dict(color='rgba(245, 158, 11, 0.75)', line=dict(color='#d97706', width=1))))
                fig_final_ops.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Real Demand"], mode='lines', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)', name=T('Procura Real', 'Real Demand'), line=dict(color='#3b82f6', width=2, shape='spline')))
                fig_final_ops.add_trace(go.Scatter(x=stockout_x, y=stockout_y, mode='markers', name=T('Stockout', 'Stockout'), marker=dict(symbol='circle', color='#ffffff', size=8, line=dict(color='#000000', width=1.5))))
                fig_final_ops.add_trace(go.Scatter(x=spoilage_x, y=spoilage_y, mode='markers', name=T('Apodrecimento', 'Spoilage'), marker=dict(symbol='diamond', color='#ef4444', size=8, line=dict(color='#991b1b', width=1.5))))
                
                fig_final_ops.update_layout(
                    title=T("Fluxo Operacional Final - RL Agent (Procura, Encomendas e Penalidades)", "Final Operational Flow - RL Agent (Demand, Orders & Penalties)"),
                    xaxis_title=T("Dias", "Days"),
                    yaxis_title=T("Quantidade (unidades)", "Quantity (units)"),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#1e293b"),
                    barmode='group',
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                )
                chart_placeholder_2.plotly_chart(fig_final_ops, use_container_width=True)

                # Plot final operacional Min-Max
                mm_stockout_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Stock Level"][idx] <= 0]
                mm_stockout_y = [plot_data["MinMax Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Stock Level"][idx] <= 0]
                
                mm_spoilage_x = [d for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Spoilage"][idx] > 0]
                mm_spoilage_y = [plot_data["MinMax Orders"][idx] for idx, d in enumerate(plot_data["Dia"]) if plot_data["MinMax Spoilage"][idx] > 0]

                fig_final_ops_mm = go.Figure()
                fig_final_ops_mm.add_trace(go.Bar(x=plot_data["Dia"], y=plot_data["MinMax Orders"], name=T('Encomendas Min-Max', 'Min-Max Orders'), marker=dict(color='rgba(140, 156, 179, 0.75)', line=dict(color='#64748b', width=1))))
                fig_final_ops_mm.add_trace(go.Scatter(x=plot_data["Dia"], y=plot_data["Real Demand"], mode='lines', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.05)', name=T('Procura Real', 'Real Demand'), line=dict(color='#3b82f6', width=2, shape='spline')))
                fig_final_ops_mm.add_trace(go.Scatter(x=mm_stockout_x, y=mm_stockout_y, mode='markers', name=T('Stockout', 'Stockout'), marker=dict(symbol='circle', color='#ffffff', size=8, line=dict(color='#000000', width=1.5))))
                fig_final_ops_mm.add_trace(go.Scatter(x=mm_spoilage_x, y=mm_spoilage_y, mode='markers', name=T('Apodrecimento', 'Spoilage'), marker=dict(symbol='diamond', color='#ef4444', size=8, line=dict(color='#991b1b', width=1.5))))
                
                fig_final_ops_mm.update_layout(
                    title=T("Fluxo Operacional Final - Min-Max (Procura, Encomendas e Penalidades)", "Final Operational Flow - Min-Max (Demand, Orders & Penalties)"),
                    xaxis_title=T("Dias", "Days"),
                    yaxis_title=T("Quantidade (unidades)", "Quantity (units)"),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#1e293b"),
                    barmode='group',
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                )
                chart_placeholder_3.plotly_chart(fig_final_ops_mm, use_container_width=True)

    if st.session_state.test_completed and st.session_state.test_results is not None:
        res = st.session_state.test_results
        plot_saved = st.session_state.get("test_plot_data", {})
        
        # --- CÁLCULO DAS MÉTRICAS DE DESPERDÍCIO / SPOILAGE ---
        spoilage_ppo = float(res.get("spoilage_total", 0.0))
        spoilage_mm = float(res.get("spoilage_minmax_total", 0.0))
        spoilage_oracle = float(res.get("spoilage_oracle_total", 0.0))
        
        orders_ppo = float(res.get("total_orders_agent", 0.0))
        orders_mm = float(res.get("total_orders_minmax", 0.0))
        orders_oracle = float(res.get("total_orders_oracle", 0.0))
        
        # Fallbacks caso venha de dados em buffer
        if orders_ppo == 0.0 and "Orders" in plot_saved and len(plot_saved["Orders"]) > 0:
            orders_ppo = float(sum(plot_saved["Orders"]))
        if orders_mm == 0.0 and "MinMax Orders" in plot_saved and len(plot_saved["MinMax Orders"]) > 0:
            orders_mm = float(sum(plot_saved["MinMax Orders"]))
        if spoilage_mm == 0.0 and "MinMax Spoilage" in plot_saved and len(plot_saved["MinMax Spoilage"]) > 0:
            spoilage_mm = float(sum(plot_saved["MinMax Spoilage"]))
            
        pct_spoilage_ppo = (spoilage_ppo / max(1.0, orders_ppo)) * 100.0 if orders_ppo > 0 else 0.0
        pct_spoilage_mm = (spoilage_mm / max(1.0, orders_mm)) * 100.0 if orders_mm > 0 else 0.0
        
        spoilage_avoided = spoilage_mm - spoilage_ppo
        if spoilage_mm > 0:
            pct_spoilage_avoided = ((spoilage_mm - spoilage_ppo) / spoilage_mm) * 100.0
        else:
            pct_spoilage_avoided = 0.0 if spoilage_ppo == 0 else -100.0

        st.markdown("---")
        st.markdown(f"### 📊 {T('Scorecard de Resultados Globais', 'Global Results Scorecard')}")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{T("Lucro Agente PPO", "PPO Agent Profit")}</div>'
                f'<div class="metric-val" style="color: #10b981;">{res["cum_profit_agent"]:,.2f}€</div>'
                f'</div>', unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{T("Lucro Min-Max", "Min-Max Profit")}</div>'
                f'<div class="metric-val" style="color: #8c9cb3;">{res["cum_profit_minmax"]:,.2f}€</div>'
                f'</div>', unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{T("Vendas Perdidas (Agente)", "Lost Sales (Agent)")}</div>'
                f'<div class="metric-val" style="color: #ffb300;">{res["lost_sales_total"]:.0f} un</div>'
                f'</div>', unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{T("Dias Stock Zero (Agente)", "Stockout Days (Agent)")}</div>'
                f'<div class="metric-val" style="color: #ef4444;">{res["stockout_days"]} {T("dias", "days")}</div>'
                f'</div>', unsafe_allow_html=True
            )

        # =====================================================================
        # SECÇÃO DE DESTAQUE: ANÁLISE COMPARATIVA DE SPOILAGE (PERECIBILIDADE)
        # =====================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🍃 {T('Eficiência de Perecibilidade & Análise de Spoilage', 'Perishability Efficiency & Spoilage Analysis')}")
        st.markdown(T(
            "Avaliação direta da capacidade do Agente RL em gerir produtos perecíveis, reduzindo quebras de validade em comparação com a política de inventário Min-Max.",
            "Direct assessment of the RL Agent's ability to manage perishable items, reducing expiration waste compared to the Min-Max inventory policy."
        ))

        col_sp1, col_sp2, col_sp3 = st.columns(3)
        with col_sp1:
            st.markdown(
                f'<div class="spoilage-standard-card">'
                f'<div class="spoilage-label">{T("Taxa Spoilage Min-Max", "Min-Max Spoilage Rate")}</div>'
                f'<div class="spoilage-val-minmax">{pct_spoilage_mm:.2f}%</div>'
                f'<div class="spoilage-sub"><b>{spoilage_mm:,.0f} un</b> {T("apodrecidas de", "spoiled out of")} {orders_mm:,.0f} un</div>'
                f'<div class="spoilage-badge badge-gray">⚙️ Baseline Min-Max</div>'
                f'</div>', unsafe_allow_html=True
            )

        with col_sp2:
            st.markdown(
                f'<div class="spoilage-standard-card" style="border-color: #a7f3d0; background: #fafffc;">'
                f'<div class="spoilage-label" style="color: #065f46;">{T("Taxa Spoilage RL Agent", "RL Agent Spoilage Rate")}</div>'
                f'<div class="spoilage-val-ppo">{pct_spoilage_ppo:.2f}%</div>'
                f'<div class="spoilage-sub"><b>{spoilage_ppo:,.0f} un</b> {T("apodrecidas de", "spoiled out of")} {orders_ppo:,.0f} un</div>'
                f'<div class="spoilage-badge badge-green">🤖 RL Agent (PPO)</div>'
                f'</div>', unsafe_allow_html=True
            )

        with col_sp3:
            if spoilage_avoided >= 0:
                st.markdown(
                    f'<div class="spoilage-hero-card">'
                    f'<div class="spoilage-label" style="color: #047857;">{T("Desperdício Evitado pelo PPO", "Spoilage Avoided by PPO")}</div>'
                    f'<div class="spoilage-val-saved">-{pct_spoilage_avoided:.1f}%</div>'
                    f'<div class="spoilage-sub"><b style="color: #065f46;">{spoilage_avoided:,.0f} un</b> {T("de fruta poupadas", "units of fruit saved")}</div>'
                    f'<div class="spoilage-badge badge-green">🏆 {T("Menos Desperdício", "Less Waste")}</div>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="spoilage-standard-card" style="border-color: #fca5a5; background: #fffbfb;">'
                    f'<div class="spoilage-label" style="color: #991b1b;">{T("Diferença de Spoilage", "Spoilage Difference")}</div>'
                    f'<div class="spoilage-val-minmax" style="color: #dc2626;">+{abs(pct_spoilage_avoided):.1f}%</div>'
                    f'<div class="spoilage-sub"><b>+{abs(spoilage_avoided):,.0f} un</b> {T("face ao Min-Max", "vs Min-Max")}</div>'
                    f'<div class="spoilage-badge badge-amber">⚠️ +{abs(spoilage_avoided):,.0f} un</div>'
                    f'</div>', unsafe_allow_html=True
                )

        # Resumo explicativo em destaque
        if spoilage_avoided > 0:
            st.success(T(
                f"🌿 **Sustentabilidade & Eficiência de Stock**: O nosso Agente PPO evitou **{pct_spoilage_avoided:.1f}%** do desperdício de fruta que ocorreria com o Min-Max tradicional, salvando **{spoilage_avoided:,.0f} unidades** de stock de apodrecer e reduzindo a taxa de quebra de **{pct_spoilage_mm:.2f}%** para **{pct_spoilage_ppo:.2f}%**.",
                f"🌿 **Sustainability & Stock Efficiency**: Our PPO Agent prevented **{pct_spoilage_avoided:.1f}%** of the fruit waste that would have occurred under traditional Min-Max, saving **{spoilage_avoided:,.0f} units** of stock from spoilage and cutting the spoilage rate from **{pct_spoilage_mm:.2f}%** down to **{pct_spoilage_ppo:.2f}%**."
            ))
        elif spoilage_avoided == 0:
            st.info(T(
                "🌿 Ambas as estratégias registaram zero quebras de apodrecimento no período testado.",
                "🌿 Both strategies registered zero spoilage events during the test period."
            ))
        else:
            st.warning(T(
                f"⚠️ O Agente PPO registou **{abs(spoilage_avoided):,.0f} unidades** a mais de desperdício face ao Min-Max (+{abs(pct_spoilage_avoided):.1f}%).",
                f"⚠️ The PPO Agent recorded **{abs(spoilage_avoided):,.0f} units** more spoilage than Min-Max (+{abs(pct_spoilage_avoided):.1f}%)."
            ))

        # --- GRÁFICOS COMPARATIVOS DE SPOILAGE ---
        col_sp_chart1, col_sp_chart2 = st.columns([7, 5])
        
        # Gráfico 1: Evolução do Spoilage Acumulado ao Longo do Tempo
        with col_sp_chart1:
            dias_series = res.get("log_dias", plot_saved.get("Dia", list(range(len(plot_saved.get("Spoilage", []))))))
            spoil_agent_daily = res.get("log_apodrecimento_agente", plot_saved.get("Spoilage", []))
            spoil_mm_daily = res.get("log_apodrecimento_minmax", plot_saved.get("MinMax Spoilage", []))
            
            cum_spoil_agent = np.cumsum(spoil_agent_daily) if len(spoil_agent_daily) > 0 else []
            cum_spoil_mm = np.cumsum(spoil_mm_daily) if len(spoil_mm_daily) > 0 else []
            
            fig_spoil_cum = go.Figure()
            if len(dias_series) > 0 and len(cum_spoil_mm) == len(dias_series):
                # Min-Max curve
                fig_spoil_cum.add_trace(go.Scatter(
                    x=dias_series,
                    y=cum_spoil_mm,
                    mode='lines',
                    name=f'Min-Max ({spoilage_mm:,.0f} un)',
                    line=dict(color='#ef4444', width=2, dash='dot')
                ))
                # RL Agent PPO curve
                fig_spoil_cum.add_trace(go.Scatter(
                    x=dias_series,
                    y=cum_spoil_agent,
                    mode='lines',
                    name=f'RL Agent PPO ({spoilage_ppo:,.0f} un)',
                    line=dict(color='#10b981', width=3),
                    fill='tonexty',
                    fillcolor='rgba(16, 185, 129, 0.12)'
                ))
            
            fig_spoil_cum.update_layout(
                title=T("Evolução do Apodrecimento Acumulado (Área = Desperdício Evitado)", "Cumulative Spoilage Evolution (Area = Prevented Waste)"),
                xaxis_title=T("Dias de Teste", "Test Days"),
                yaxis_title=T("Unidades Apodrecidas Acumuladas", "Cumulative Spoiled Units"),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#1e293b"),
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                margin=dict(t=50, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_spoil_cum, use_container_width=True)

        # Gráfico 2: Comparação de Taxas e Volumes de Spoilage
        with col_sp_chart2:
            categories = [T("RL Agent (PPO)", "RL Agent (PPO)"), T("Min-Max Baseline", "Min-Max Baseline")]
            rates = [pct_spoilage_ppo, pct_spoilage_mm]
            units = [spoilage_ppo, spoilage_mm]
            colors = ['#10b981', '#ef4444']
            
            fig_spoil_bar = go.Figure()
            fig_spoil_bar.add_trace(go.Bar(
                x=categories,
                y=rates,
                name=T("Taxa de Spoilage (%)", "Spoilage Rate (%)"),
                marker_color=colors,
                text=[f"{r:.2f}%<br>({u:,.0f} un)" for r, u in zip(rates, units)],
                textposition='auto',
                width=0.45
            ))
            
            fig_spoil_bar.update_layout(
                title=T("Taxa de Spoilage Comparativa (%)", "Comparative Spoilage Rate (%)"),
                yaxis_title=T("Taxa de Apodrecimento (%)", "Spoilage Rate (%)"),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#1e293b"),
                margin=dict(t=50, b=40, l=40, r=40),
                showlegend=False
            )
            st.plotly_chart(fig_spoil_bar, use_container_width=True)

        # Comparações diretas de lucro
        profit_diff = res["cum_profit_agent"] - res["cum_profit_minmax"]
        pct_improvement = (profit_diff / max(1.0, abs(res["cum_profit_minmax"]))) * 100.0
        
        st.markdown(f"#### {T('Resumo Financeiro & Operacional', 'Financial & Operational Summary')}")
        if profit_diff > 0:
            st.success(T(f"📈 O Agente PPO superou o baseline Min-Max tradicional em **{profit_diff:.2f}€** (+{pct_improvement:.2f}% de lucro).", f"📈 The PPO Agent outperformed the traditional Min-Max baseline by **{profit_diff:.2f}€** (+{pct_improvement:.2f}% profit)."))
        else:
            st.warning(T(f"📉 O Agente PPO obteve um lucro inferior ao baseline Min-Max tradicional em **{abs(profit_diff):.2f}€** ({pct_improvement:.2f}%).", f"📉 The PPO Agent obtained a profit lower than the traditional Min-Max baseline by **{abs(profit_diff):.2f}€** ({pct_improvement:.2f}%)."))
            
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown(f"#### {T('📥 Exportar Resultados e Modelo Otimizado', '📥 Export Results & Optimized Model')}")
        col_down1, col_down2 = st.columns(2)
        
        with col_down1:
            # Downloader do excel de simulação
            if os.path.exists(res["excel_report_path"]):
                with open(res["excel_report_path"], "rb") as f:
                    st.download_button(
                        label=T("📊 Descarregar Relatório Excel (.xlsx)", "📊 Download Excel Report (.xlsx)"),
                        data=f.read(),
                        file_name="relatorio_simulacao_buyer_agent.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
        with col_down2:
            # Downloader do modelo fine-tuned final
            zip_final_bytes = create_model_zip_bytes(res["final_model_path"])
            if zip_final_bytes is not None:
                st.download_button(
                    label=T("📥 Descarregar Modelo Otimizado / Fine-Tuned (.ZIP)", "📥 Download Optimized / Fine-Tuned Model (.ZIP)"),
                    data=zip_final_bytes,
                    file_name="modelo_ppo_buyer_agent_fine_tuned.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        st.markdown('</div>', unsafe_allow_html=True)
