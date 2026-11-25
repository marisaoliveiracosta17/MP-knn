# Atividade: o LLM resolve melhor que o nosso modelo? (urgência em Medida Protetiva)

Tarefa: classificar o texto do relato em `urgente` ou `nao_urgente`, sem usar o FONAR.
Entrega: link do notebook + `.zip` com `resultados_oficina/` no Moodle, até 23h59 de 25/11/2026.
Demonstração: 3 minutos. Vale 20% da nota.

## 1. Arquivos desta pasta
| Arquivo | Para quê |
|---|---|
| `conjunto_teste.csv` | 30 exemplos com `saida_esperada`, `saida_modelo_proprio` e `confianca_modelo_proprio` |
| `treino_modelo_proprio.csv` | Treino do modelo próprio (não entra no teste nem no few-shot) |
| `exemplos_few_shot.csv` | 8 exemplos para a v3 |
| `prompts.py` | `Saida`, `FALLBACK`, `TAREFA_V1`, `TAREFA_V2` |
| `gerar_conjuntos.py` | Regenera os CSVs e confere que não há vazamento |
| `requirements.txt` | Dependências Python |
| `README.md` | Critério de rótulo e limitações |

| `oficina_notebook_urgencia.ipynb` | Notebook-base do professor já adaptado ao caso (esquema, prompts v1/v2/v3, few-shot, metas) |

O notebook original está em https://github.com/bmnogueira-ufms/LLM-ESIA-2026 (pasta `oficina/`).
No notebook adaptado, as Seções 3 e 4 já estão preenchidas. Falta só ajustar `GRUPO` e as metas da Seção 1, escolher a opção A ou B e rodar.
Suba para o Colab junto com `conjunto_teste.csv` e `exemplos_few_shot.csv`.

## 2. Preparar o ambiente no PC
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gerar_conjuntos.py        # opcional: regenera os CSVs
```
Para rodar o notebook localmente, instale também `jupyter`. No Colab, basta enviar os CSVs.

## 3. Escolher o acesso ao LLM
- **Opção A (Colab, sem custo):** `Qwen/Qwen2.5-1.5B-Instruct` na GPU T4. Os dados não saem do Colab.
- **Opção B (API compatível com OpenAI):** guarde a chave em `LLM_API_KEY` (Secrets do Colab ou variável de ambiente). Nunca no código ou no notebook.
- **Ollama local:** `BASE_URL = "http://localhost:11434/v1"`.
Rode a Seção 0 do notebook e confirme que imprime resposta, tokens e latência.

## 4. Configurar o notebook
Seção 1:
```python
GRUPO = "Grupo X"
MVP = "triagem de medidas protetivas: classifica urgência pelo texto do relato"
TAREFA = "classificacao"
META_METRICA = 0.85        # ajuste para a meta da sua ficha
META_LATENCIA_S = 2.0
META_CUSTO_MES_USD = 20.0
CHAMADAS_POR_MES = 30_000  # volume esperado
```
Seção 2: envie `conjunto_teste.csv` para a pasta do Colab. Deve aparecer
"comparação com o modelo próprio: sim" e "híbrido por confiança: sim".

Seção 3 (cole de `prompts.py`): `Saida` e `FALLBACK`. A categoria é
`Literal["urgente", "nao_urgente", "indefinido"]`.

Seção 4: use `TAREFA_V1` na v1. Para o few-shot:
```python
import pandas as pd
EXEMPLOS_FEW_SHOT = [(e, {"categoria": s}) for e, s in pd.read_csv("exemplos_few_shot.csv").values]
```

## 5. Passo a passo
1. **Sprint 1:** rode `rodar("v1")` e `detalhar("v1")`. Poste o checkpoint:
   `Grupo X · v1 · acurácia __% · inválidas __% · p95 __ s · US$ __/mês`.
2. **Sprint 2:** crie a v2 com `TAREFA_V2`. Depois a v3 com a v2 mais few-shot. Mude uma coisa por versão e anote em `nota`.
3. Use `TEMPERATURA = 0` e rode duas vezes para conferir estabilidade.
4. Compare com o modelo próprio e varie `LIMIAR` no híbrido.
5. Abra `erros_<versao>.csv` e preencha `tipo_erro` (ambíguo, rótulo errado, fora do formato, faltou contexto, alucinação, erro grosseiro).
6. Preencha `DECISAO_PROVISORIA` ("modelo próprio", "LLM" ou "híbrido") e `JUSTIFICATIVA` com números.
7. Rode a última célula e baixe o `.zip`. Baixe também ao fim de cada sprint.

## 6. O que olhar nos resultados
- Revocação de `urgente`: um urgente classificado como nao_urgente é o pior erro.
- F1 macro, saídas inválidas, latência p50/p95 e custo por mês.
- Com 30 exemplos, um acerto vale cerca de 3 pontos percentuais. Não decida por diferença de 1 ou 2 acertos.
- Casos de ataque (ids 16 e 30): o LLM obedeceu ao texto injetado?

## 7. Cuidados
- Nenhum exemplo do teste pode estar no treino nem no few-shot. O `gerar_conjuntos.py` confere isso.
- Os textos são sintéticos. Os rótulos foram definidos por critério textual, não por servidor da triagem. Se possível, peça a revisão dos 30 rótulos.
- As regras da `TAREFA_V2` foram escritas olhando estes exemplos. Ajuste-as com base nos erros, sem copiar frases do teste.
- Dados reais de relatos têm dados pessoais: mascare antes de enviar a API externa.
- Declare o uso de IA generativa no documento final.

## 8. Checklist de entrega (Moodle)
- [ ] link do notebook com permissão de leitura
- [ ] `.zip` com `registro_experimentos.csv`, `resultados_<versao>.csv`, `quadro_comparacao.csv`, `erros_<versao>.csv` com `tipo_erro`
- [ ] pelo menos 2 versões com nota do que mudou
- [ ] demonstração em até 3 minutos: exemplo (acerto e erro), quadro e decisão com número
