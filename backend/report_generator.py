"""
Detailed Report Generator
Generates comprehensive research reports with figures and tables
Uses PI Agent to plan and Bioinformatics Agent to execute code
"""
import os
import json
import uuid
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from .models.schemas import AnalysisState, AgentType
from .utils.llm_client import OpenAICompatibleClient
from .utils.sandbox import LocalJupyterSandbox

logger = logging.getLogger(__name__)


class DetailedReportGenerator:
    """
    Generates detailed research reports based on analysis results.
    
    Workflow:
    1. PI Agent plans the report structure based on analysis conclusions
    2. For each section, PI Agent gives specific instructions to Bioinformatics Agent
    3. Bioinformatics Agent writes and executes code to generate figures/tables
    4. Results are saved as PNG/PDF (figures) and CSV (tables)
    5. PI Agent writes the final report narrative
    """
    
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        pi_model: str,
        bio_model: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ):
        self.llm_client = llm_client
        self.pi_model = pi_model
        self.bio_model = bio_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        
        # Create subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        # Jupyter sandbox for code execution
        self.sandbox: Optional[LocalJupyterSandbox] = None
        
    async def _report_progress(self, progress: float, step: str, status: str = "generating"):
        """Report progress to callback"""
        if self.progress_callback:
            await self.progress_callback({
                "progress": progress,
                "step": step,
                "status": status
            })
    
    async def _pi_chat(self, prompt: str, temperature: float = 0.3) -> str:
        """Chat with PI Agent"""
        response = await self.llm_client.chat(
            model=self.pi_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.content if response else ""
    
    async def _bio_chat(self, prompt: str, temperature: float = 0.2) -> str:
        """Chat with Bioinformatics Agent"""
        response = await self.llm_client.chat(
            model=self.bio_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.content if response else ""
    
    async def generate_report(
        self,
        state: AnalysisState,
    ) -> Dict[str, Any]:
        """
        Generate a detailed research report
        
        Args:
            state: Analysis state with results
            
        Returns:
            Detailed report structure
        """
        report_id = str(uuid.uuid4())
        language = "zh" if any('\u4e00' <= c <= '\u9fff' for c in state.initial_instruction) else "en"
        
        await self._report_progress(2, "Initializing..." if language == "en" else "初始化...")
        
        # Initialize Jupyter sandbox
        self.sandbox = LocalJupyterSandbox(session_id=f"report_{report_id}")
        await self.sandbox.start()
        
        try:
            # Setup the analysis environment
            await self._report_progress(5, "Setting up environment..." if language == "en" else "设置分析环境...")
            await self._setup_environment(state)
            
            # Extract conclusions from analysis
            await self._report_progress(8, "Extracting conclusions..." if language == "en" else "提取分析结论...")
            conclusions = await self._extract_conclusions(state)
            
            # PI Agent plans the report
            await self._report_progress(10, "Planning report structure..." if language == "en" else "规划报告结构...")
            report_plan = await self._plan_report(state, conclusions, language)
            
            # Generate each section
            sections = []
            total_sections = len(report_plan.get("sections", []))
            
            for i, section_plan in enumerate(report_plan.get("sections", [])):
                progress = 15 + (i / max(total_sections, 1)) * 70
                section_title = section_plan.get("title", f"Section {i+1}")
                await self._report_progress(
                    progress, 
                    f"Generating: {section_title}" if language == "en" else f"生成: {section_title}"
                )
                
                section = await self._generate_section(
                    state, 
                    section_plan, 
                    conclusions,
                    language
                )
                sections.append(section)
            
            # PI Agent writes the report narrative
            await self._report_progress(90, "Writing report..." if language == "en" else "撰写报告...")
            abstract = await self._generate_abstract(state, sections, language)
            final_conclusions = await self._generate_final_conclusions(state, sections, conclusions, language)
            
            report = {
                "id": report_id,
                "session_id": state.id,
                "title": report_plan.get("title", state.name),
                "abstract": abstract,
                "sections": sections,
                "conclusions": final_conclusions,
                "generated_at": datetime.now().isoformat(),
                "status": "completed",
                "progress": 100,
                "current_step": "Complete" if language == "en" else "完成"
            }
            
            # Save report metadata
            report_path = self.output_dir / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            await self._report_progress(100, "Complete" if language == "en" else "完成", "completed")
            
            return report
            
        finally:
            # Cleanup sandbox
            if self.sandbox:
                await self.sandbox.stop()
    
    async def _setup_environment(self, state: AnalysisState):
        """Setup the Jupyter environment with data and analysis packages"""
        
        # Build comprehensive setup code
        setup_code = f"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Initialize adata as None (will be loaded later)
adata = None

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.max_open_warning'] = 50

# Output directories
FIGURES_DIR = "{self.figures_dir}"
TABLES_DIR = "{self.tables_dir}"

# ============================================================
# Check available packages
# ============================================================

# Cell communication packages
LIANA_AVAILABLE = False
CELLCHAT_AVAILABLE = False
SQUIDPY_AVAILABLE = False
COMMOT_AVAILABLE = False

try:
    import liana as li
    LIANA_AVAILABLE = True
    print("✓ LIANA available")
except ImportError:
    print("✗ LIANA not available")

try:
    import squidpy as sq
    SQUIDPY_AVAILABLE = True
    print("✓ Squidpy available")
except ImportError:
    print("✗ Squidpy not available")

# Pathway enrichment packages
GSEAPY_AVAILABLE = False
DECOUPLER_AVAILABLE = False

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
    print("✓ GSEApy available")
except ImportError:
    print("✗ GSEApy not available")

try:
    import decoupler as dc
    DECOUPLER_AVAILABLE = True
    print("✓ decoupler available")
except ImportError:
    print("✗ decoupler not available")

# Trajectory analysis packages
SCVELO_AVAILABLE = False
PAGA_AVAILABLE = True  # Part of scanpy

try:
    import scvelo as scv
    SCVELO_AVAILABLE = True
    print("✓ scVelo available")
except ImportError:
    print("✗ scVelo not available")

# ============================================================
# Helper functions
# ============================================================

def save_figure(fig, filepath_base, dpi=300):
    \"\"\"Save figure as both PNG and PDF\"\"\"
    png_path = f"{{filepath_base}}.png"
    pdf_path = f"{{filepath_base}}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Figure saved: {{png_path}}")
    return png_path, pdf_path

def get_cell_type_key(adata):
    \"\"\"Find the cell type annotation column\"\"\"
    for key in ['cell_type', 'celltype', 'CellType', 'cell_types', 'leiden', 'louvain', 'clusters']:
        if key in adata.obs.columns:
            return key
    return None

def get_group_key(adata):
    \"\"\"Find the experimental group column\"\"\"
    for key in ['group', 'condition', 'sample', 'batch', 'Group', 'Condition']:
        if key in adata.obs.columns:
            return key
    return None

def safe_rank_genes_groups(adata, groupby, groups='all', reference='rest', method='wilcoxon'):
    \"\"\"Safely run rank_genes_groups with error handling\"\"\"
    try:
        sc.tl.rank_genes_groups(adata, groupby=groupby, groups=groups, 
                                reference=reference, method=method, use_raw=False)
        return True
    except Exception as e:
        print(f"rank_genes_groups failed: {{e}}")
        return False

# ============================================================
# Cell Communication Analysis Functions
# ============================================================

def run_liana_analysis(adata, cell_type_key):
    \"\"\"Run LIANA cell communication analysis\"\"\"
    if not LIANA_AVAILABLE:
        return None
    try:
        # Run LIANA with multiple methods
        li.mt.rank_aggregate(
            adata,
            groupby=cell_type_key,
            resource_name='consensus',
            expr_prop=0.1,
            verbose=True,
            use_raw=False
        )
        return adata.uns['liana_res']
    except Exception as e:
        print(f"LIANA analysis failed: {{e}}")
        return None

def run_squidpy_ligrec(adata, cell_type_key):
    \"\"\"Run Squidpy ligand-receptor analysis\"\"\"
    if not SQUIDPY_AVAILABLE:
        return None
    try:
        sq.gr.ligrec(
            adata,
            n_perms=100,
            cluster_key=cell_type_key,
            copy=False,
            use_raw=False,
            transmitter_params={{'categories': 'ligand'}},
            receiver_params={{'categories': 'receptor'}}
        )
        return adata.uns[f'{{cell_type_key}}_ligrec']
    except Exception as e:
        print(f"Squidpy ligrec failed: {{e}}")
        return None

# Comprehensive ligand-receptor database (expanded)
LR_DATABASE = [
    # TGF-beta signaling
    ('TGFB1', 'TGFBR1'), ('TGFB1', 'TGFBR2'), ('TGFB2', 'TGFBR1'), ('TGFB3', 'TGFBR2'),
    # TNF signaling
    ('TNF', 'TNFRSF1A'), ('TNF', 'TNFRSF1B'), ('TNFSF10', 'TNFRSF10A'), ('TNFSF10', 'TNFRSF10B'),
    ('LTA', 'TNFRSF1A'), ('LTB', 'LTBR'), ('FASLG', 'FAS'),
    # Interleukin signaling
    ('IL1A', 'IL1R1'), ('IL1B', 'IL1R1'), ('IL1B', 'IL1R2'),
    ('IL2', 'IL2RA'), ('IL2', 'IL2RB'), ('IL4', 'IL4R'), ('IL6', 'IL6R'), ('IL6', 'IL6ST'),
    ('IL10', 'IL10RA'), ('IL10', 'IL10RB'), ('IL12A', 'IL12RB1'), ('IL12B', 'IL12RB2'),
    ('IL13', 'IL13RA1'), ('IL15', 'IL15RA'), ('IL17A', 'IL17RA'), ('IL18', 'IL18R1'),
    ('IL21', 'IL21R'), ('IL23A', 'IL23R'), ('IL33', 'IL1RL1'),
    # Chemokine signaling
    ('CXCL1', 'CXCR2'), ('CXCL2', 'CXCR2'), ('CXCL8', 'CXCR1'), ('CXCL8', 'CXCR2'),
    ('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'), ('CXCL11', 'CXCR3'),
    ('CXCL12', 'CXCR4'), ('CXCL12', 'CXCR7'), ('CXCL13', 'CXCR5'),
    ('CCL2', 'CCR2'), ('CCL3', 'CCR1'), ('CCL3', 'CCR5'), ('CCL4', 'CCR5'),
    ('CCL5', 'CCR1'), ('CCL5', 'CCR3'), ('CCL5', 'CCR5'),
    ('CCL19', 'CCR7'), ('CCL20', 'CCR6'), ('CCL21', 'CCR7'),
    ('CX3CL1', 'CX3CR1'),
    # Growth factors
    ('VEGFA', 'FLT1'), ('VEGFA', 'KDR'), ('VEGFA', 'NRP1'), ('VEGFB', 'FLT1'),
    ('VEGFC', 'FLT4'), ('VEGFD', 'FLT4'),
    ('PDGFA', 'PDGFRA'), ('PDGFB', 'PDGFRB'), ('PDGFC', 'PDGFRA'), ('PDGFD', 'PDGFRB'),
    ('EGF', 'EGFR'), ('HBEGF', 'EGFR'), ('AREG', 'EGFR'), ('EREG', 'EGFR'),
    ('FGF1', 'FGFR1'), ('FGF2', 'FGFR1'), ('FGF7', 'FGFR2'), ('FGF10', 'FGFR2'),
    ('HGF', 'MET'), ('IGF1', 'IGF1R'), ('IGF2', 'IGF1R'), ('IGF2', 'IGF2R'),
    ('NGF', 'NTRK1'), ('BDNF', 'NTRK2'), ('NTF3', 'NTRK3'),
    # BMP signaling
    ('BMP2', 'BMPR1A'), ('BMP2', 'BMPR1B'), ('BMP4', 'BMPR1A'), ('BMP4', 'BMPR2'),
    ('BMP6', 'ACVR1'), ('BMP7', 'BMPR1B'), ('GDF15', 'GFRAL'),
    # WNT signaling
    ('WNT1', 'FZD1'), ('WNT2', 'FZD5'), ('WNT3', 'FZD1'), ('WNT3A', 'FZD1'),
    ('WNT4', 'FZD2'), ('WNT5A', 'FZD5'), ('WNT5A', 'ROR2'), ('WNT5B', 'FZD5'),
    ('WNT7A', 'FZD5'), ('WNT7B', 'FZD1'), ('WNT11', 'FZD7'),
    ('RSPO1', 'LGR4'), ('RSPO2', 'LGR5'), ('RSPO3', 'LGR4'),
    # Notch signaling
    ('DLL1', 'NOTCH1'), ('DLL1', 'NOTCH2'), ('DLL3', 'NOTCH1'), ('DLL4', 'NOTCH1'),
    ('JAG1', 'NOTCH1'), ('JAG1', 'NOTCH2'), ('JAG2', 'NOTCH1'), ('JAG2', 'NOTCH3'),
    # Hedgehog signaling
    ('SHH', 'PTCH1'), ('SHH', 'PTCH2'), ('IHH', 'PTCH1'), ('DHH', 'PTCH1'),
    # ECM interactions
    ('COL1A1', 'ITGA1'), ('COL1A1', 'ITGA2'), ('COL1A1', 'ITGB1'), ('COL1A1', 'DDR1'),
    ('COL1A2', 'ITGA1'), ('COL1A2', 'ITGB1'), ('COL4A1', 'ITGA1'), ('COL4A1', 'ITGB1'),
    ('FN1', 'ITGA5'), ('FN1', 'ITGB1'), ('FN1', 'ITGAV'), ('FN1', 'SDC4'),
    ('LAMA1', 'ITGA6'), ('LAMA1', 'ITGB1'), ('LAMB1', 'ITGA6'),
    ('VTN', 'ITGAV'), ('VTN', 'ITGB3'),
    ('THBS1', 'CD36'), ('THBS1', 'CD47'), ('THBS1', 'ITGAV'), ('THBS2', 'CD36'),
    ('SPP1', 'CD44'), ('SPP1', 'ITGAV'), ('SPP1', 'ITGB1'), ('SPP1', 'ITGB3'),
    # Semaphorin signaling
    ('SEMA3A', 'NRP1'), ('SEMA3A', 'PLXNA1'), ('SEMA3C', 'NRP1'), ('SEMA3F', 'NRP2'),
    ('SEMA4D', 'PLXNB1'), ('SEMA4D', 'CD72'), ('SEMA7A', 'ITGB1'),
    # Ephrin signaling
    ('EFNA1', 'EPHA2'), ('EFNA1', 'EPHA4'), ('EFNB1', 'EPHB1'), ('EFNB1', 'EPHB2'),
    ('EFNB2', 'EPHB4'), ('EFNA5', 'EPHA5'),
    # Immune checkpoints
    ('CD274', 'PDCD1'), ('PDCD1LG2', 'PDCD1'), ('CD80', 'CD28'), ('CD80', 'CTLA4'),
    ('CD86', 'CD28'), ('CD86', 'CTLA4'), ('LGALS9', 'HAVCR2'), ('TNFRSF9', 'TNFSF9'),
    ('ICOS', 'ICOSLG'), ('CD40', 'CD40LG'), ('CD70', 'CD27'),
    # Other important pairs
    ('ANGPT1', 'TEK'), ('ANGPT2', 'TEK'), ('ANGPTL4', 'ITGB1'),
    ('GAS6', 'AXL'), ('GAS6', 'MERTK'), ('PROS1', 'AXL'),
    ('APOE', 'LDLR'), ('APOE', 'LRP1'), ('APP', 'LRP1'),
    ('CSF1', 'CSF1R'), ('CSF2', 'CSF2RA'), ('CSF3', 'CSF3R'),
    ('KIT', 'KITLG'), ('FLT3', 'FLT3LG'),
    ('MIF', 'CD74'), ('MIF', 'CXCR4'), ('HMGB1', 'AGER'), ('HMGB1', 'TLR4'),
    ('CALR', 'LRP1'), ('ANXA1', 'FPR1'), ('ANXA1', 'FPR2'),
]

def comprehensive_lr_analysis(adata, cell_type_key='cell_type', min_expr=0.1, min_pct=0.1):
    \"\"\"
    Comprehensive ligand-receptor analysis using built-in database
    \"\"\"
    if cell_type_key not in adata.obs.columns:
        print(f"Cell type key '{{cell_type_key}}' not found")
        return pd.DataFrame()
    
    cell_types = adata.obs[cell_type_key].unique().tolist()
    print(f"Analyzing {{len(cell_types)}} cell types: {{cell_types}}")
    
    # Calculate mean expression and percentage for each gene in each cell type
    gene_stats = {{}}
    for ct in cell_types:
        mask = adata.obs[cell_type_key] == ct
        subset = adata[mask]
        n_cells = subset.n_obs
        
        gene_stats[ct] = {{}}
        for gene in adata.var_names:
            if gene in [lr[0] for lr in LR_DATABASE] + [lr[1] for lr in LR_DATABASE]:
                expr = subset[:, gene].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                else:
                    expr = np.array(expr).flatten()
                
                mean_expr = np.mean(expr)
                pct_expr = np.sum(expr > 0) / n_cells if n_cells > 0 else 0
                gene_stats[ct][gene] = {{'mean': mean_expr, 'pct': pct_expr}}
    
    # Find significant interactions
    interactions = []
    for ligand, receptor in LR_DATABASE:
        if ligand not in adata.var_names or receptor not in adata.var_names:
            continue
        
        for source in cell_types:
            for target in cell_types:
                lig_stat = gene_stats[source].get(ligand, {{'mean': 0, 'pct': 0}})
                rec_stat = gene_stats[target].get(receptor, {{'mean': 0, 'pct': 0}})
                
                if (lig_stat['mean'] >= min_expr and lig_stat['pct'] >= min_pct and
                    rec_stat['mean'] >= min_expr and rec_stat['pct'] >= min_pct):
                    
                    score = lig_stat['mean'] * rec_stat['mean']
                    interactions.append({{
                        'source': source,
                        'target': target,
                        'ligand': ligand,
                        'receptor': receptor,
                        'ligand_mean': lig_stat['mean'],
                        'ligand_pct': lig_stat['pct'],
                        'receptor_mean': rec_stat['mean'],
                        'receptor_pct': rec_stat['pct'],
                        'lr_pair': f"{{ligand}}_{{receptor}}",
                        'score': score
                    }})
    
    df = pd.DataFrame(interactions)
    if len(df) > 0:
        df = df.sort_values('score', ascending=False)
        print(f"Found {{len(df)}} significant interactions")
    return df

# ============================================================
# Pathway Enrichment Functions
# ============================================================

def run_enrichr(gene_list, gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'], organism='human'):
    \"\"\"Run Enrichr analysis using GSEApy\"\"\"
    if not GSEAPY_AVAILABLE or len(gene_list) == 0:
        return None
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            no_plot=True
        )
        return enr.results
    except Exception as e:
        print(f"Enrichr failed: {{e}}")
        return None

def run_gsea_prerank(gene_scores, gene_sets='GO_Biological_Process_2021'):
    \"\"\"Run GSEA prerank analysis\"\"\"
    if not GSEAPY_AVAILABLE:
        return None
    try:
        pre_res = gp.prerank(
            rnk=gene_scores,
            gene_sets=gene_sets,
            processes=4,
            permutation_num=100,
            outdir=None,
            no_plot=True
        )
        return pre_res.res2d
    except Exception as e:
        print(f"GSEA prerank failed: {{e}}")
        return None

def run_decoupler_pathway(adata, cell_type_key=None):
    \"\"\"Run decoupler pathway activity inference\"\"\"
    if not DECOUPLER_AVAILABLE:
        return None
    try:
        # Get PROGENy model
        progeny = dc.get_progeny(organism='human', top=500)
        
        # Run mlm
        dc.run_mlm(
            mat=adata,
            net=progeny,
            source='source',
            target='target',
            weight='weight',
            verbose=True,
            use_raw=False
        )
        
        return adata.obsm['mlm_estimate']
    except Exception as e:
        print(f"decoupler pathway failed: {{e}}")
        return None

print("\\n" + "="*60)
print("Environment setup complete")
print(f"Figures directory: {{FIGURES_DIR}}")
print(f"Tables directory: {{TABLES_DIR}}")
print(f"adata initialized: {{adata is not None}}")
print("="*60)
"""
        
        result = await self.sandbox.execute(setup_code)
        if result.error:
            logger.warning(f"Setup warning: {result.error}")
        
        # Debug: log the paths
        logger.info(f"Data paths - control: {state.control_seq_path}, treatment: {state.treatment_seq_path}")
        
        # Load data if available
        if state.control_seq_path or state.treatment_seq_path:
            load_code = "# Load scRNA-seq data\n"
            
            if state.control_seq_path and state.treatment_seq_path:
                load_code += f"""
import os
ctrl_path = "{state.control_seq_path}"
treat_path = "{state.treatment_seq_path}"

print(f"Control path: {{ctrl_path}}, exists: {{os.path.exists(ctrl_path)}}")
print(f"Treatment path: {{treat_path}}, exists: {{os.path.exists(treat_path)}}")

if os.path.exists(ctrl_path) and os.path.exists(treat_path):
    adata_control = sc.read_h5ad(ctrl_path)
    adata_control.obs['group'] = 'Control'
    print(f"Control: {{adata_control.n_obs}} cells, {{adata_control.n_vars}} genes")
    
    adata_treatment = sc.read_h5ad(treat_path)
    adata_treatment.obs['group'] = 'Treatment'
    print(f"Treatment: {{adata_treatment.n_obs}} cells, {{adata_treatment.n_vars}} genes")
    
    # Merge datasets
    adata = sc.concat([adata_control, adata_treatment], join='outer')
    print(f"Merged: {{adata.n_obs}} cells, {{adata.n_vars}} genes")
    print(f"Obs columns: {{list(adata.obs.columns)}}")
elif os.path.exists(ctrl_path):
    adata = sc.read_h5ad(ctrl_path)
    print(f"Loaded control only: {{adata.n_obs}} cells")
elif os.path.exists(treat_path):
    adata = sc.read_h5ad(treat_path)
    print(f"Loaded treatment only: {{adata.n_obs}} cells")
else:
    print("ERROR: Data files not found!")

print(f"\\n>>> DATA LOADING COMPLETE: adata is {{'loaded' if adata is not None else 'None'}}")
"""
            elif state.control_seq_path:
                load_code += f"""
import os
path = "{state.control_seq_path}"
print(f"Control path: {{path}}, exists: {{os.path.exists(path)}}")
if os.path.exists(path):
    adata = sc.read_h5ad(path)
    print(f"Loaded: {{adata.n_obs}} cells, {{adata.n_vars}} genes")
else:
    print("ERROR: Control data file not found!")

print(f"\\n>>> DATA LOADING COMPLETE: adata is {{'loaded' if adata is not None else 'None'}}")
"""
            elif state.treatment_seq_path:
                load_code += f"""
import os
path = "{state.treatment_seq_path}"
print(f"Treatment path: {{path}}, exists: {{os.path.exists(path)}}")
if os.path.exists(path):
    adata = sc.read_h5ad(path)
    print(f"Loaded: {{adata.n_obs}} cells, {{adata.n_vars}} genes")
else:
    print("ERROR: Treatment data file not found!")

print(f"\\n>>> DATA LOADING COMPLETE: adata is {{'loaded' if adata is not None else 'None'}}")
"""
            
            result = await self.sandbox.execute(load_code)
            logger.info(f"Data loading result: {result.output[:500] if result.output else 'No output'}")
            if result.error:
                logger.warning(f"Data loading error: {result.error[:500]}")
        else:
            # No data paths - initialize adata as None
            logger.warning("No data paths provided! control_seq_path and treatment_seq_path are both None")
            init_code = "print('WARNING: No data paths provided, adata remains None')"
            await self.sandbox.execute(init_code)
    
    async def _extract_conclusions(self, state: AnalysisState) -> List[str]:
        """Extract key conclusions from the analysis"""
        
        all_findings = []
        for round_result in state.round_results:
            if round_result.pathology_findings:
                all_findings.extend(round_result.pathology_findings)
            if round_result.bioinformatics_findings:
                all_findings.extend(round_result.bioinformatics_findings)
            if round_result.biochemistry_findings:
                all_findings.extend(round_result.biochemistry_findings)
        
        prompt = f"""
Based on the analysis findings, extract 5-10 key scientific conclusions:

## Research Goal
{state.initial_instruction}

## Final Summary
{state.final_summary or "Not available"}

## Findings
{json.dumps(all_findings[:30], ensure_ascii=False, indent=2)}

Return as JSON array of strings:
["conclusion1", "conclusion2", ...]
"""
        
        response = await self._pi_chat(prompt)
        
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return all_findings[:10] if all_findings else ["General characterization"]
    
    async def _plan_report(
        self, 
        state: AnalysisState, 
        conclusions: List[str],
        language: str
    ) -> Dict[str, Any]:
        """PI Agent plans the report structure with comprehensive analyses"""
        
        zh = language == "zh"
        
        # Comprehensive analysis plan
        return {
            "title": "单细胞RNA测序综合分析报告" if zh else "Single-Cell RNA Sequencing Analysis Report",
            "sections": [
                # ============================================================
                # 1. Quality Control
                # ============================================================
                {
                    "id": "qc",
                    "title": "质量控制" if zh else "Quality Control",
                    "description": "数据质量评估和细胞过滤" if zh else "Data quality assessment and filtering",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "QC小提琴图" if zh else "QC Violin Plot",
                            "instruction": "绘制小提琴图显示每个样本的n_genes_by_counts, total_counts, pct_counts_mt分布，按group分面显示" if zh else "Plot violin plots for n_genes_by_counts, total_counts, pct_counts_mt per sample, faceted by group",
                        },
                        {
                            "type": "figure",
                            "name": "QC散点图" if zh else "QC Scatter Plot",
                            "instruction": "绘制散点图显示total_counts vs n_genes_by_counts，按pct_counts_mt着色，识别异常细胞" if zh else "Plot scatter of total_counts vs n_genes_by_counts, colored by pct_counts_mt to identify outliers",
                        },
                        {
                            "type": "figure",
                            "name": "基因表达分布" if zh else "Gene Expression Distribution",
                            "instruction": "绘制直方图显示每个基因在细胞中的表达频率分布，以及每个细胞的基因数分布" if zh else "Plot histogram of gene detection frequency and genes per cell distribution",
                        },
                        {
                            "type": "table",
                            "name": "QC统计表" if zh else "QC Statistics Table",
                            "instruction": "统计每个样本/组的细胞数量、中位基因数、中位UMI数、中位线粒体比例" if zh else "Statistics of cell count, median genes, median UMIs, median mito% per sample/group",
                        }
                    ]
                },
                
                # ============================================================
                # 2. Cell Clustering
                # ============================================================
                {
                    "id": "clustering",
                    "title": "细胞聚类分析" if zh else "Cell Clustering",
                    "description": "UMAP降维和聚类结果" if zh else "UMAP dimensionality reduction and clustering",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "UMAP聚类图" if zh else "UMAP Clusters",
                            "instruction": "绘制UMAP图，按leiden/louvain聚类着色，显示聚类标签" if zh else "Plot UMAP colored by leiden/louvain clusters with labels",
                        },
                        {
                            "type": "figure",
                            "name": "UMAP分组图" if zh else "UMAP by Group",
                            "instruction": "绘制UMAP图，按实验组(group)着色，并绘制分面图展示每组的分布" if zh else "Plot UMAP colored by experimental group, with split view for each group",
                        },
                        {
                            "type": "figure",
                            "name": "UMAP样本分布" if zh else "UMAP by Sample",
                            "instruction": "绘制UMAP图，按样本batch着色（如果有多个样本）" if zh else "Plot UMAP colored by sample/batch if multiple samples exist",
                        },
                        {
                            "type": "figure",
                            "name": "聚类树状图" if zh else "Cluster Dendrogram",
                            "instruction": "绘制聚类之间的层次树状图，显示聚类之间的关系" if zh else "Plot dendrogram showing hierarchical relationship between clusters",
                        },
                        {
                            "type": "table",
                            "name": "聚类统计" if zh else "Cluster Statistics",
                            "instruction": "统计每个聚类的细胞数量和在各组中的分布" if zh else "Statistics of cell counts per cluster and distribution across groups",
                        }
                    ]
                },
                
                # ============================================================
                # 3. Cell Type Annotation
                # ============================================================
                {
                    "id": "annotation",
                    "title": "细胞类型注释" if zh else "Cell Type Annotation",
                    "description": "细胞类型鉴定和marker基因" if zh else "Cell type identification and markers",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "Marker基因热图" if zh else "Marker Heatmap",
                            "instruction": "绘制热图显示每个聚类的top 5 marker基因表达" if zh else "Plot heatmap of top 5 marker genes per cluster",
                        },
                        {
                            "type": "figure",
                            "name": "Marker基因点图" if zh else "Marker Dotplot",
                            "instruction": "绘制dotplot显示已知细胞类型marker基因在各聚类中的表达（百分比和平均表达）" if zh else "Plot dotplot of known cell type markers showing expression percentage and mean expression per cluster",
                        },
                        {
                            "type": "figure",
                            "name": "细胞类型UMAP" if zh else "Cell Type UMAP",
                            "instruction": "绘制UMAP图，按注释的细胞类型(cell_type)着色" if zh else "Plot UMAP colored by annotated cell_type",
                        },
                        {
                            "type": "figure",
                            "name": "Marker基因特征图" if zh else "Marker Feature Plot",
                            "instruction": "绘制UMAP特征图显示4-6个关键marker基因的表达分布" if zh else "Plot UMAP feature plots showing expression of 4-6 key marker genes",
                        },
                        {
                            "type": "table",
                            "name": "聚类Marker基因" if zh else "Cluster Markers",
                            "instruction": "输出每个聚类的top 20 marker基因列表，包括log2FC, pct.1, pct.2, p_val_adj" if zh else "Export top 20 marker genes per cluster with log2FC, pct.1, pct.2, p_val_adj",
                        }
                    ]
                },
                
                # ============================================================
                # 4. Cell Proportion Analysis
                # ============================================================
                {
                    "id": "proportion",
                    "title": "细胞比例分析" if zh else "Cell Proportion Analysis",
                    "description": "各组细胞类型比例对比" if zh else "Cell type proportion comparison between groups",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "细胞比例堆叠柱状图" if zh else "Stacked Proportion Bar",
                            "instruction": "绘制100%堆叠柱状图显示各组的细胞类型比例" if zh else "Plot 100% stacked bar chart of cell type proportions by group",
                        },
                        {
                            "type": "figure",
                            "name": "细胞比例分组柱状图" if zh else "Grouped Proportion Bar",
                            "instruction": "绘制分组柱状图比较各细胞类型在不同组中的比例，添加误差线（如果有重复样本）" if zh else "Plot grouped bar chart comparing cell type proportions between groups with error bars if replicates exist",
                        },
                        {
                            "type": "figure",
                            "name": "细胞数量热图" if zh else "Cell Count Heatmap",
                            "instruction": "绘制热图显示各样本中各细胞类型的数量" if zh else "Plot heatmap of cell counts per cell type per sample",
                        },
                        {
                            "type": "table",
                            "name": "细胞比例统计表" if zh else "Proportion Statistics Table",
                            "instruction": "统计各细胞类型在每组的细胞数、比例、以及组间比例变化的统计检验结果" if zh else "Statistics of cell counts, proportions per group, and statistical test results for proportion changes",
                        }
                    ]
                },
                
                # ============================================================
                # 5. Differential Expression Analysis
                # ============================================================
                {
                    "id": "de",
                    "title": "差异表达分析" if zh else "Differential Expression Analysis",
                    "description": "组间差异基因分析" if zh else "Differential expression between groups",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "整体差异基因火山图" if zh else "Overall Volcano Plot",
                            "instruction": "绘制所有细胞的Treatment vs Control差异基因火山图，标注显著上调和下调基因的数量和代表性基因名" if zh else "Plot volcano plot of all cells Treatment vs Control, annotate significant up/down regulated gene counts and representative gene names",
                        },
                        {
                            "type": "figure",
                            "name": "分细胞类型火山图" if zh else "Cell Type Volcano Plots",
                            "instruction": "绘制分面火山图，每个细胞类型单独显示差异基因（选择主要细胞类型）" if zh else "Plot faceted volcano plots for each major cell type showing DE genes",
                        },
                        {
                            "type": "figure",
                            "name": "差异基因热图" if zh else "DE Heatmap",
                            "instruction": "绘制热图显示top 50差异基因在各组各细胞类型中的表达，按细胞类型和组别排序" if zh else "Plot heatmap of top 50 DE genes expression across groups and cell types, sorted by cell type and group",
                        },
                        {
                            "type": "figure",
                            "name": "差异基因数量柱状图" if zh else "DE Gene Counts Bar",
                            "instruction": "绘制柱状图显示每个细胞类型的上调和下调基因数量" if zh else "Plot bar chart showing up and down regulated gene counts per cell type",
                        },
                        {
                            "type": "figure",
                            "name": "关键差异基因箱线图" if zh else "Key DE Gene Boxplot",
                            "instruction": "绘制top 6差异基因在各组各细胞类型中的表达箱线图" if zh else "Plot boxplots of top 6 DE genes expression across groups and cell types",
                        },
                        {
                            "type": "table",
                            "name": "整体差异基因列表" if zh else "Overall DE Gene List",
                            "instruction": "输出所有细胞的完整差异基因列表，包括gene, log2FC, pct.1, pct.2, p_val, p_val_adj" if zh else "Export full DE gene list for all cells with gene, log2FC, pct.1, pct.2, p_val, p_val_adj",
                        },
                        {
                            "type": "table",
                            "name": "分细胞类型差异基因" if zh else "Cell Type DE Genes",
                            "instruction": "输出每个细胞类型的top差异基因，合并为一个表格" if zh else "Export top DE genes per cell type combined into one table",
                        }
                    ]
                },
                
                # ============================================================
                # 6. Pathway Enrichment Analysis
                # ============================================================
                {
                    "id": "pathway",
                    "title": "通路富集分析" if zh else "Pathway Enrichment Analysis",
                    "description": "GO和KEGG通路富集分析" if zh else "GO and KEGG pathway enrichment analysis",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "GO生物过程富集" if zh else "GO Biological Process",
                            "instruction": "使用上调差异基因进行GO Biological Process富集分析，绘制top 15通路的条形图，显示-log10(p-value)和基因数" if zh else "Run GO Biological Process enrichment on upregulated DE genes, plot bar chart of top 15 pathways with -log10(p-value) and gene counts",
                        },
                        {
                            "type": "figure",
                            "name": "KEGG通路富集" if zh else "KEGG Pathway Enrichment",
                            "instruction": "使用差异基因进行KEGG通路富集分析，绘制top 15通路的气泡图，显示富集分数、基因数和p值" if zh else "Run KEGG pathway enrichment on DE genes, plot bubble chart of top 15 pathways with enrichment score, gene count and p-value",
                        },
                        {
                            "type": "figure",
                            "name": "分细胞类型通路富集" if zh else "Cell Type Pathway Enrichment",
                            "instruction": "绘制热图显示各细胞类型中富集的关键通路（GO/KEGG）" if zh else "Plot heatmap showing enriched pathways (GO/KEGG) per cell type",
                        },
                        {
                            "type": "figure",
                            "name": "GSEA富集图" if zh else "GSEA Enrichment Plot",
                            "instruction": "使用预排序的基因列表（按log2FC）进行GSEA分析，绘制top富集通路的GSEA曲线" if zh else "Run GSEA with preranked genes (by log2FC), plot GSEA curves for top enriched pathways",
                        },
                        {
                            "type": "table",
                            "name": "GO富集结果" if zh else "GO Enrichment Results",
                            "instruction": "输出GO富集分析完整结果，包括Term, Overlap, P-value, Adjusted P-value, Genes" if zh else "Export full GO enrichment results with Term, Overlap, P-value, Adjusted P-value, Genes",
                        },
                        {
                            "type": "table",
                            "name": "KEGG富集结果" if zh else "KEGG Enrichment Results",
                            "instruction": "输出KEGG富集分析完整结果" if zh else "Export full KEGG enrichment results",
                        }
                    ]
                },
                
                # ============================================================
                # 7. Cell-Cell Communication Analysis
                # ============================================================
                {
                    "id": "communication",
                    "title": "细胞通讯分析" if zh else "Cell-Cell Communication Analysis",
                    "description": "配体-受体相互作用和细胞间通讯网络" if zh else "Ligand-receptor interactions and cell-cell communication networks",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "通讯数量热图" if zh else "Communication Count Heatmap",
                            "instruction": "使用LIANA或comprehensive_lr_analysis分析细胞通讯，绘制热图显示各细胞类型对之间的相互作用数量" if zh else "Analyze cell communication using LIANA or comprehensive_lr_analysis, plot heatmap of interaction counts between cell type pairs",
                        },
                        {
                            "type": "figure",
                            "name": "通讯强度热图" if zh else "Communication Strength Heatmap",
                            "instruction": "绘制热图显示各细胞类型对之间的相互作用强度（通讯得分总和）" if zh else "Plot heatmap of interaction strength (sum of communication scores) between cell type pairs",
                        },
                        {
                            "type": "figure",
                            "name": "通讯圈图" if zh else "Communication Circle Plot",
                            "instruction": "绘制圈图(chord diagram)显示主要的细胞间通讯关系，线条粗细代表相互作用强度" if zh else "Plot chord diagram showing major cell-cell communication relationships, line width represents interaction strength",
                        },
                        {
                            "type": "figure",
                            "name": "配体-受体气泡图" if zh else "Ligand-Receptor Dotplot",
                            "instruction": "绘制气泡图显示top 30配体-受体对在各细胞类型对中的表达，气泡大小代表得分，颜色代表表达水平" if zh else "Plot dotplot of top 30 ligand-receptor pairs across cell type pairs, bubble size for score, color for expression level",
                        },
                        {
                            "type": "figure",
                            "name": "信号通路活性" if zh else "Signaling Pathway Activity",
                            "instruction": "按信号通路分类汇总配体-受体对，绘制热图显示各通路在不同细胞类型作为发送者/接收者时的活性" if zh else "Aggregate L-R pairs by signaling pathway, plot heatmap of pathway activity for each cell type as sender/receiver",
                        },
                        {
                            "type": "figure",
                            "name": "组间通讯差异" if zh else "Communication Difference Between Groups",
                            "instruction": "如果有两组数据，分别计算通讯并绘制差异热图，显示Treatment vs Control的通讯变化" if zh else "If two groups exist, calculate communication separately and plot difference heatmap showing Treatment vs Control changes",
                        },
                        {
                            "type": "figure",
                            "name": "特定细胞类型通讯" if zh else "Specific Cell Type Communication",
                            "instruction": "选择1-2个关键细胞类型，详细展示其作为发送者和接收者时的主要通讯对象和配体-受体对" if zh else "Select 1-2 key cell types, show detailed communication partners and L-R pairs as sender and receiver",
                        },
                        {
                            "type": "table",
                            "name": "配体-受体完整列表" if zh else "Full Ligand-Receptor List",
                            "instruction": "输出完整的配体-受体相互作用列表，包括source, target, ligand, receptor, ligand_expr, receptor_expr, score, pathway" if zh else "Export full L-R interaction list with source, target, ligand, receptor, ligand_expr, receptor_expr, score, pathway",
                        },
                        {
                            "type": "table",
                            "name": "通路汇总表" if zh else "Pathway Summary Table",
                            "instruction": "按信号通路汇总相互作用数量和强度" if zh else "Summarize interaction counts and strength by signaling pathway",
                        }
                    ]
                },
                
                # ============================================================
                # 8. Trajectory Analysis (if applicable)
                # ============================================================
                {
                    "id": "trajectory",
                    "title": "拟时序分析" if zh else "Trajectory Analysis",
                    "description": "细胞分化轨迹和拟时序分析" if zh else "Cell differentiation trajectory and pseudotime analysis",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "PAGA图" if zh else "PAGA Graph",
                            "instruction": "使用PAGA分析细胞群之间的连接关系，绘制PAGA图叠加在UMAP上" if zh else "Run PAGA analysis for cluster connectivity, plot PAGA graph overlaid on UMAP",
                        },
                        {
                            "type": "figure",
                            "name": "拟时序UMAP" if zh else "Pseudotime UMAP",
                            "instruction": "使用diffusion pseudotime或其他方法计算拟时序，绘制UMAP按拟时序着色" if zh else "Calculate pseudotime using diffusion pseudotime or similar, plot UMAP colored by pseudotime",
                        },
                        {
                            "type": "figure",
                            "name": "轨迹基因热图" if zh else "Trajectory Gene Heatmap",
                            "instruction": "绘制热图显示沿拟时序变化的基因表达模式，识别早期和晚期基因" if zh else "Plot heatmap of gene expression changes along pseudotime, identify early and late genes",
                        },
                        {
                            "type": "table",
                            "name": "轨迹相关基因" if zh else "Trajectory-Associated Genes",
                            "instruction": "输出与拟时序显著相关的基因列表" if zh else "Export list of genes significantly associated with pseudotime",
                        }
                    ]
                },
                
                # ============================================================
                # 9. Targeted Analysis
                # ============================================================
                {
                    "id": "targeted",
                    "title": "针对性分析" if zh else "Targeted Analysis",
                    "description": "基于研究结论的特定分析" if zh else "Specific analyses based on research conclusions",
                    "analyses": [
                        {
                            "type": "figure",
                            "name": "结论验证图1" if zh else "Conclusion Validation 1",
                            "instruction": f"基于研究结论进行针对性分析: {conclusions[0] if conclusions else '关键基因表达分析'}。绘制相关基因的表达图或细胞类型特异性分析" if zh else f"Targeted analysis based on conclusion: {conclusions[0] if conclusions else 'key gene expression'}. Plot relevant gene expression or cell type specific analysis",
                        },
                        {
                            "type": "figure",
                            "name": "结论验证图2" if zh else "Conclusion Validation 2",
                            "instruction": f"基于研究结论进行针对性分析: {conclusions[1] if len(conclusions) > 1 else '差异通路验证'}。" if zh else f"Targeted analysis based on conclusion: {conclusions[1] if len(conclusions) > 1 else 'differential pathway validation'}.",
                        },
                        {
                            "type": "figure",
                            "name": "综合结论图" if zh else "Summary Conclusion Figure",
                            "instruction": "绘制一个综合图展示研究的主要发现，可以是多面板图整合关键结果" if zh else "Create a summary figure showing main findings, can be multi-panel integrating key results",
                        }
                    ]
                }
            ]
        }
    
    async def _generate_section(
        self,
        state: AnalysisState,
        section_plan: Dict[str, Any],
        conclusions: List[str],
        language: str
    ) -> Dict[str, Any]:
        """Generate a single report section"""
        
        section_id = section_plan.get("id", str(uuid.uuid4()))
        section_title = section_plan.get("title", "Untitled")
        
        figures = []
        tables = []
        
        for analysis in section_plan.get("analyses", []):
            analysis_type = analysis.get("type", "figure")
            analysis_name = analysis.get("name", "Unnamed")
            instruction = analysis.get("instruction", "")
            
            logger.info(f"Generating {analysis_type}: {analysis_name}")
            
            if analysis_type == "figure":
                figure = await self._generate_figure(
                    analysis_name, instruction, section_id, language
                )
                if figure:
                    figures.append(figure)
                    
            elif analysis_type == "table":
                table = await self._generate_table(
                    analysis_name, instruction, section_id, language
                )
                if table:
                    tables.append(table)
        
        # Write section narrative
        section_content = await self._write_section_narrative(
            section_title, section_plan.get("description", ""),
            figures, tables, language
        )
        
        return {
            "id": section_id,
            "title": section_title,
            "content": section_content,
            "figures": figures,
            "tables": tables
        }
    
    async def _generate_figure(
        self,
        name: str,
        instruction: str,
        section_id: str,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a figure using Bioinformatics Agent"""
        
        figure_id = f"{section_id}_{name.replace(' ', '_').lower()}"
        figure_id = re.sub(r'[^a-z0-9_]', '', figure_id.lower())
        figure_base = str(self.figures_dir / figure_id)
        png_path = f"{figure_base}.png"
        pdf_path = f"{figure_base}.pdf"
        
        # Bioinformatics Agent generates code
        code_prompt = f"""
Write Python code to generate a figure for single-cell RNA-seq analysis.

## Task
{instruction}

## Environment
The following are already available:
- adata: AnnData object with scRNA-seq data (check if not None)
- sc (scanpy), plt (matplotlib.pyplot), sns (seaborn), np (numpy), pd (pandas)
- Helper functions: save_figure(fig, filepath_base), get_cell_type_key(adata), get_group_key(adata)
- Cell communication: comprehensive_lr_analysis(adata, cell_type_key), run_liana_analysis(adata, cell_type_key)
- Pathway analysis: run_enrichr(gene_list), run_gsea_prerank(gene_scores), run_decoupler_pathway(adata)
- Booleans: LIANA_AVAILABLE, SQUIDPY_AVAILABLE, GSEAPY_AVAILABLE, DECOUPLER_AVAILABLE, SCVELO_AVAILABLE

## Common obs columns
- Cell type: 'cell_type', 'celltype', 'leiden', 'louvain'
- Group: 'group', 'condition', 'sample'
- QC metrics: 'n_genes_by_counts', 'total_counts', 'pct_counts_mt'

## Requirements
1. Always check if adata is not None before proceeding
2. Use get_cell_type_key(adata) and get_group_key(adata) to find correct column names
3. Handle missing data gracefully with try-except
4. Save figure using: save_figure(fig, "{figure_base}")
5. Create publication-quality figures with proper labels, titles, and legends
6. Use figsize appropriate for the plot type (e.g., 10x8 for UMAP, 12x10 for heatmaps)

## Example Templates

### For UMAP plots:
```python
if adata is not None:
    cell_type_key = get_cell_type_key(adata)
    if cell_type_key:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color=cell_type_key, ax=ax, show=False, 
                   legend_loc='on data', legend_fontsize=8)
        ax.set_title('UMAP by Cell Type')
        save_figure(fig, "{figure_base}")
```

### For heatmaps:
```python
if adata is not None:
    # Get top marker genes
    if 'rank_genes_groups' in adata.uns:
        fig = sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, 
                                               show_gene_labels=True, show=False)
        save_figure(plt.gcf(), "{figure_base}")
```

### For cell communication:
```python
if adata is not None:
    cell_type_key = get_cell_type_key(adata)
    if cell_type_key:
        lr_df = comprehensive_lr_analysis(adata, cell_type_key=cell_type_key)
        if len(lr_df) > 0:
            # Create heatmap of interaction counts
            counts = lr_df.groupby(['source', 'target']).size().reset_index(name='count')
            pivot = counts.pivot(index='source', columns='target', values='count').fillna(0)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(pivot, annot=True, fmt='g', cmap='Reds', ax=ax)
            ax.set_title('Cell-Cell Communication Counts')
            save_figure(fig, "{figure_base}")
```

### For pathway enrichment:
```python
if adata is not None and GSEAPY_AVAILABLE:
    # Get DE genes
    if 'rank_genes_groups' in adata.uns:
        de_genes = sc.get.rank_genes_groups_df(adata, group=None)
        up_genes = de_genes[de_genes['logfoldchanges'] > 0.5]['names'].tolist()[:200]
        
        # Run enrichment
        enr_results = run_enrichr(up_genes, gene_sets=['GO_Biological_Process_2021'])
        if enr_results is not None and len(enr_results) > 0:
            # Plot top terms
            top_terms = enr_results.head(15)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_terms)), -np.log10(top_terms['Adjusted P-value']))
            ax.set_yticks(range(len(top_terms)))
            ax.set_yticklabels(top_terms['Term'])
            ax.set_xlabel('-log10(Adjusted P-value)')
            ax.set_title('GO Biological Process Enrichment')
            plt.tight_layout()
            save_figure(fig, "{figure_base}")
```

### For volcano plots:
```python
if adata is not None:
    group_key = get_group_key(adata)
    if group_key and 'Treatment' in adata.obs[group_key].values:
        sc.tl.rank_genes_groups(adata, groupby=group_key, groups=['Treatment'], 
                                reference='Control', method='wilcoxon')
        de_df = sc.get.rank_genes_groups_df(adata, group='Treatment')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(de_df['logfoldchanges'], -np.log10(de_df['pvals_adj']), 
                   c='gray', alpha=0.5, s=10)
        # Highlight significant genes
        sig = (de_df['pvals_adj'] < 0.05) & (abs(de_df['logfoldchanges']) > 1)
        ax.scatter(de_df.loc[sig, 'logfoldchanges'], -np.log10(de_df.loc[sig, 'pvals_adj']), 
                   c='red', s=20)
        ax.axhline(-np.log10(0.05), color='gray', linestyle='--')
        ax.axvline(-1, color='gray', linestyle='--')
        ax.axvline(1, color='gray', linestyle='--')
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('-log10(Adjusted P-value)')
        ax.set_title('Volcano Plot')
        save_figure(fig, "{figure_base}")
```

Write complete, executable Python code for the specific task. Be robust to missing data.
"""
        
        code_response = await self._bio_chat(code_prompt)
        code = self._extract_code(code_response)
        
        if not code:
            logger.warning(f"No code generated for figure: {name}")
            return None
        
        # Execute code
        result = await self.sandbox.execute(code)
        output = result.output or ""
        error = result.error or ""
        
        logger.info(f"Figure {name} output: {output[:300]}")
        if error:
            logger.warning(f"Figure {name} error: {error[:300]}")
        
        # Check if files were created
        if not os.path.exists(png_path):
            logger.warning(f"Figure not created: {name}")
            return None
        
        # Generate description
        description = await self._describe_result("figure", name, instruction, output, language)
        
        return {
            "id": figure_id,
            "title": name,
            "description": description.get("description", instruction),
            "png_path": png_path,
            "pdf_path": pdf_path if os.path.exists(pdf_path) else "",
            "analysis_method": description.get("method", instruction),
            "results": description.get("results", "See figure"),
            "conclusion": description.get("conclusion", "")
        }
    
    async def _generate_table(
        self,
        name: str,
        instruction: str,
        section_id: str,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a table using Bioinformatics Agent"""
        
        table_id = f"{section_id}_{name.replace(' ', '_').lower()}"
        table_id = re.sub(r'[^a-z0-9_]', '', table_id.lower())
        csv_path = str(self.tables_dir / f"{table_id}.csv")
        
        # Bioinformatics Agent generates code
        code_prompt = f"""
Write Python code to create a table for single-cell RNA-seq analysis.

## Task
{instruction}

## Environment
The following are already available:
- adata: AnnData object with scRNA-seq data (check if not None)
- sc (scanpy), plt (matplotlib.pyplot), sns (seaborn), np (numpy), pd (pandas)
- Helper functions: get_cell_type_key(adata), get_group_key(adata)
- Cell communication: comprehensive_lr_analysis(adata, cell_type_key)
- Pathway analysis: run_enrichr(gene_list)
- Booleans: LIANA_AVAILABLE, GSEAPY_AVAILABLE, DECOUPLER_AVAILABLE

## Requirements
1. Always check if adata is not None before proceeding
2. Create a pandas DataFrame with meaningful results
3. Save to CSV: df.to_csv("{csv_path}", index=False) or index=True if index is meaningful
4. Print summary and first few rows
5. Handle missing data gracefully

## Example Templates

### For QC statistics:
```python
if adata is not None:
    group_key = get_group_key(adata)
    stats = []
    groups = adata.obs[group_key].unique() if group_key else ['All']
    for g in groups:
        mask = adata.obs[group_key] == g if group_key else [True]*adata.n_obs
        subset = adata[mask]
        stats.append({{
            'Group': g,
            'n_cells': subset.n_obs,
            'median_genes': subset.obs['n_genes_by_counts'].median() if 'n_genes_by_counts' in subset.obs else np.nan,
            'median_counts': subset.obs['total_counts'].median() if 'total_counts' in subset.obs else np.nan,
            'median_mito_pct': subset.obs['pct_counts_mt'].median() if 'pct_counts_mt' in subset.obs else np.nan
        }})
    df = pd.DataFrame(stats)
    df.to_csv("{csv_path}", index=False)
    print(df)
```

### For cell proportion statistics:
```python
if adata is not None:
    cell_type_key = get_cell_type_key(adata)
    group_key = get_group_key(adata)
    if cell_type_key:
        # Cross-tabulation
        ct = pd.crosstab(adata.obs[cell_type_key], 
                         adata.obs[group_key] if group_key else 'All',
                         margins=True)
        # Add percentages
        ct_pct = ct.div(ct.iloc[:, :-1].sum(axis=0), axis=1) * 100
        ct_pct = ct_pct.round(2)
        ct_pct.columns = [f'{{c}}_pct' for c in ct_pct.columns]
        result = pd.concat([ct, ct_pct], axis=1)
        result.to_csv("{csv_path}")
        print(result)
```

### For DE gene list:
```python
if adata is not None and 'rank_genes_groups' in adata.uns:
    de_df = sc.get.rank_genes_groups_df(adata, group=None)
    de_df = de_df.sort_values('pvals_adj')
    de_df.to_csv("{csv_path}", index=False)
    print(f"Total DE genes: {{len(de_df)}}")
    print(f"Significant (padj<0.05): {{(de_df['pvals_adj'] < 0.05).sum()}}")
    print(de_df.head(20))
```

### For cell communication:
```python
if adata is not None:
    cell_type_key = get_cell_type_key(adata)
    if cell_type_key:
        lr_df = comprehensive_lr_analysis(adata, cell_type_key=cell_type_key)
        if len(lr_df) > 0:
            lr_df = lr_df.sort_values('score', ascending=False)
            lr_df.to_csv("{csv_path}", index=False)
            print(f"Total interactions: {{len(lr_df)}}")
            print(lr_df.head(20))
```

### For pathway enrichment:
```python
if adata is not None and GSEAPY_AVAILABLE:
    if 'rank_genes_groups' in adata.uns:
        de_df = sc.get.rank_genes_groups_df(adata, group=None)
        up_genes = de_df[de_df['logfoldchanges'] > 0.5]['names'].tolist()[:300]
        
        enr = run_enrichr(up_genes, gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'])
        if enr is not None:
            enr.to_csv("{csv_path}", index=False)
            print(f"Enriched terms: {{len(enr)}}")
            print(enr[['Term', 'Adjusted P-value', 'Overlap']].head(20))
```

Write complete, executable Python code for the specific task. Be robust to missing data.
"""
        
        code_response = await self._bio_chat(code_prompt)
        code = self._extract_code(code_response)
        
        if not code:
            logger.warning(f"No code generated for table: {name}")
            return None
        
        # Execute code
        result = await self.sandbox.execute(code)
        output = result.output or ""
        error = result.error or ""
        
        logger.info(f"Table {name} output: {output[:300]}")
        if error:
            logger.warning(f"Table {name} error: {error[:300]}")
        
        # Read the CSV
        columns = []
        preview_data = []
        
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                columns = df.columns.tolist()
                preview_data = df.head(20).to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Failed to read table: {e}")
        else:
            logger.warning(f"Table not created: {name}")
            return None
        
        # Generate description
        description = await self._describe_result("table", name, instruction, output, language)
        
        return {
            "id": table_id,
            "title": name,
            "description": description.get("description", instruction),
            "csv_path": csv_path,
            "columns": columns,
            "preview_data": preview_data,
            "analysis_method": description.get("method", instruction),
            "results": description.get("results", "See table"),
            "conclusion": description.get("conclusion", "")
        }
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code block
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if code_match:
            return code_match.group(1).strip()
        
        code_match = re.search(r'```\s*([\s\S]*?)\s*```', response)
        if code_match:
            return code_match.group(1).strip()
        
        # Return the whole response if it looks like code
        if any(kw in response for kw in ['import', 'adata', 'plt.', 'sc.', 'pd.', 'df']):
            return response.strip()
        
        return ""
    
    async def _describe_result(
        self,
        result_type: str,
        name: str,
        instruction: str,
        output: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate description for a figure or table"""
        
        prompt = f"""
Describe this {result_type} for a scientific report (2-3 sentences each):

Name: {name}
Analysis: {instruction}
Output: {output[:500] if output else "Generated successfully"}

Return JSON:
{{"description": "...", "method": "...", "results": "...", "conclusion": "..."}}

Language: {"Chinese" if language == "zh" else "English"}
"""
        
        response = await self._pi_chat(prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"description": instruction, "method": instruction, "results": "See result", "conclusion": ""}
    
    async def _write_section_narrative(
        self,
        section_title: str,
        description: str,
        figures: List[Dict],
        tables: List[Dict],
        language: str
    ) -> str:
        """Write narrative text for a section"""
        
        fig_info = [f"- {f['title']}: {f.get('results', '')[:100]}" for f in figures]
        tab_info = [f"- {t['title']}: {len(t.get('preview_data', []))} rows" for t in tables]
        
        prompt = f"""
Write 2-3 paragraphs for this report section:

Section: {section_title}
Description: {description}

Figures: {chr(10).join(fig_info) if fig_info else "None"}
Tables: {chr(10).join(tab_info) if tab_info else "None"}

Language: {"Chinese" if language == "zh" else "English"}
"""
        
        response = await self._pi_chat(prompt)
        return response.strip()
    
    async def _generate_abstract(
        self,
        state: AnalysisState,
        sections: List[Dict],
        language: str
    ) -> str:
        """Generate report abstract"""
        
        section_info = [f"- {s['title']}: {len(s.get('figures', []))} figures, {len(s.get('tables', []))} tables" for s in sections]
        
        prompt = f"""
Write a 200-300 word abstract:

Research Goal: {state.initial_instruction}
Summary: {state.final_summary or "Analysis completed"}
Sections: {chr(10).join(section_info)}

Language: {"Chinese" if language == "zh" else "English"}
"""
        
        response = await self._pi_chat(prompt)
        return response.strip()
    
    async def _generate_final_conclusions(
        self,
        state: AnalysisState,
        sections: List[Dict],
        conclusions: List[str],
        language: str
    ) -> str:
        """Generate final conclusions"""
        
        prompt = f"""
Write 3-5 paragraphs of conclusions:

Research Goal: {state.initial_instruction}
Key Conclusions: {json.dumps(conclusions[:5], ensure_ascii=False)}
Summary: {state.final_summary or "Analysis completed"}

Include: main findings, biological significance, limitations, future directions.
Language: {"Chinese" if language == "zh" else "English"}
"""
        
        response = await self._pi_chat(prompt)
        return response.strip()
