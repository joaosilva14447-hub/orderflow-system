# daily_signals — Bottom Radar

Pipeline diário e autónomo que lê **4 pilares** ortogonais do mercado crypto,
junta-os num **Accumulation Score 0–100** com setup nomeado, guarda histórico
versionado e avisa no Discord. Corre sozinho via GitHub Actions — zero trabalho
manual, zero análise manual.

## Os 4 pilares

| Pilar | Módulo | Mede | Fonte (grátis, sem chave*) |
|---|---|---|---|
| 🌊 **Doom** | [doom_index.py](doom_index.py) | Capitulação narrativa (pânico nas notícias) | RSS + Claude Haiku* |
| 📈 **Valuation (SDCA)** | [valuation.py](valuation.py) | Quão barato (MVRV Z, Mayer, Puell, Metcalfe) | Coin Metrics |
| 🌐 **Macro** | [macro.py](macro.py) | Regime de liquidez (net liquidity, dólar, crédito) | FRED |
| 🔗 **Posicionamento** | [positioning.py](positioning.py) | Dry powder + funding (stablecoins, SSR, funding) | DefiLlama + Binance |

\* Só o Doom usa uma chave (Anthropic, ~€0,50/mês). Sem ela, cai num fallback de
keywords. Os outros 3 pilares são 100% gratuitos e sem chave.

## O composto

[composite.py](composite.py) orienta cada pilar para "favorece acumulação"
(barato + pânico + macro a favor + carregado = alto) e faz a **média com pesos
iguais** → o Accumulation Score. Mas o número é só para relance: o **setup
nomeado** e as **flags de alinhamento/divergência** carregam o sinal real, para
não esconder conflitos entre pilares (ex. "barato mas macro hostil").

O **gate** dispara o alerta forte 🚨 quando há **capitulação (Doom alto/extremo)
COM valuation barato** — o sinal de bottom que realmente interessa.

## Metodologia (anti-overfit)

- **Sem lookahead:** normalização por percentil rolling de ~4 anos, só com dados
  passados.
- **Pesos iguais, thresholds redondos:** nada é otimizado no backtest.
- **Diversidade ortogonal:** um indicador por família; nada de contar a mesma
  coisa duas vezes.
- **Doom** começa `warming_up` (precisa de ~30 dias para o percentil); os outros
  pilares nascem calibrados (anos de histórico).

## Storage

- [history/doom_index.csv](history) — uma linha/dia com composto + os 4 pilares +
  todos os sub-indicadores. É a série backtestável.
- `history/headlines/AAAA-MM-DD.json` — texto cru das manchetes (o teu arquivo
  pesquisável, que nenhuma API vende retroativamente).

## Política de envio (anti-fadiga)

Envia no Discord quando: há setup forte/gate, a zona do Doom ou do Valuation
**muda**, o Doom está `high`/`extreme`, ou é **domingo** (digest). Força com
`FORCE_SEND=1`.

## Setup (uma vez)

1. **Secrets** (Settings → Secrets and variables → Actions):
   - `ANTHROPIC_API_KEY` — key da Anthropic (Doom scoring, ~€0,50/mês).
   - `DISCORD_WEBHOOK_URL` — webhook do canal.
2. O workflow [daily-doom.yml](../.github/workflows/daily-doom.yml) corre às
   **08:05 UTC** e faz commit do snapshot. Corre à mão em Actions → *Run workflow*.

## Correr localmente (Windows)

```powershell
pip install -r daily_signals/requirements.txt
cd daily_signals
$env:PYTHONIOENCODING="utf-8"
$env:ANTHROPIC_API_KEY="sk-ant-..."      # opcional
$env:DISCORD_WEBHOOK_URL="https://..."   # opcional
$env:FORCE_SEND="1"
python run.py
```

## Personalizar

- **Fontes de notícias:** [feeds.yaml](feeds.yaml).
- **Indicadores / janelas:** constantes no topo de cada módulo de pilar.
- **Zonas e setups:** `_zone()` em cada pilar; `_named_setup()`/`_label()` em
  [composite.py](composite.py).
