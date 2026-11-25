# Oficina LLM: urgência em Medida Protetiva (sem FONAR)

Tarefa: classificar o texto de `descricao_fatos` em `urgente` ou `nao_urgente`
quando nenhuma situação de risco alto (FONAR) foi marcada.

## Critério de rótulo (definir na ficha)
**urgente**: o relato descreve risco atual ou iminente à vida ou à integridade:
ameaça de morte, arma, agressão grave (estrangulamento, facada, socos com lesão),
violência sexual, cárcere, perseguição com ameaça, descumprimento de medida
anterior com invasão ou violência, ameaça a filhos ou familiares.
**nao_urgente**: violência psicológica, moral ou patrimonial sem risco imediato,
conflito sem ameaça, fato antigo sem contato posterior, sem lesão ou sem arma.
Regra de fronteira: na dúvida real entre as duas, vale `urgente` (errar para
cima custa menos que atrasar um caso grave).

## Arquivos
| Arquivo | Uso |
|---|---|
| `conjunto_teste.csv` | 30 exemplos (16 urgente / 14 nao_urgente), 2 ataques de prompt injection, colunas `saida_modelo_proprio` e `confianca_modelo_proprio` |
| `treino_modelo_proprio.csv` | 48 exemplos usados só no modelo próprio |
| `exemplos_few_shot.csv` | 8 exemplos (4/4, 4 de fronteira) para v2/v3 |
| `gerar_conjuntos.py` | Regenera tudo e confere que os três conjuntos não se sobrepõem |

Os textos são sintéticos, sem dados pessoais. O modelo próprio de referência é
TF-IDF + regressão logística treinado só no treino. Troque pelo modelo do grupo
mantendo as mesmas colunas.

## Configuração do notebook-base
```python
TAREFA = "classificacao"
class Saida(BaseModel):
    categoria: Literal["urgente", "nao_urgente", "indefinido"]
    justificativa: str = Field(default="", max_length=300)
FALLBACK = {"categoria": "urgente"}  # na falha, tratar como urgente
EXEMPLOS_FEW_SHOT = [(e, {"categoria": s}) for e, s in pd.read_csv("exemplos_few_shot.csv").values]
```
Métrica sugerida: F1 macro e, principalmente, a **revocação de `urgente`**
(um urgente classificado como nao_urgente é o pior erro).

## Limitações
- O rótulo foi definido por critério textual do grupo, não por servidor da triagem.
  Para o documento final, peça a um servidor para revisar os 30 rótulos.
- Com 30 exemplos, um acerto vale cerca de 3 pontos percentuais.
- Dados reais de `descricao_fatos` têm dados pessoais: mascare antes de enviar a API externa.

## Prompts
`prompts.py` tem `Saida`, `FALLBACK`, `TAREFA_V1` (zero-shot) e `TAREFA_V2`
(descrições, regras de fronteira, justificativa antes da categoria). Cole nos
`# TODO` das Seções 3 e 4 do notebook. A v3 é a v2 mais `EXEMPLOS_FEW_SHOT`.
Mude uma coisa por versão e anote em `nota`.
