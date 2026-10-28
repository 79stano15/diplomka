# Hybridné kvantovo-klasické reinforcement learning pre kockovú hru

## O projekte

Projekt skúma, či sa parametrizovaný kvantový obvod dá prakticky začleniť do
policy pre reinforcement learning a ako sa jeho správanie a náklady porovnávajú
s klasickými sieťami. Kocková hra je kontrolovateľný proof of concept: má
presne definované pravidlá, náhodné hody aj opakované strategické rozhodnutia.
Nie je cieľom tvrdiť, že experiment preukazuje kvantovú výhodu alebo že hra je
finálnym aplikačným use case.

V hre hrajú dvaja hráči so šiestimi kockami do cieľového skóre 4 000. Po hode
policy vyberá spomedzi legálnych bodujúcich kombinácií kociek; podľa akcie môže
body zapísať a odovzdať ťah, alebo pokračovať v hode. Pravidlá zahŕňajú aj
neúspešný hod bez bodujúcej kocky a opätovné získanie všetkých kociek pri
využití celého hodu.

Policy skóruje dvojicu **stav hry – legálna akcia**. Každá dvojica je
reprezentovaná 22 príznakmi a policy vytvorí jedno skóre pre každú dostupnú
akciu; počas tréningu sa z týchto skóre vzorkuje, pri greedy evaluácii sa
vyberie najvyššie skóre. Tréning používa odmenu podľa výsledku zápasu a
baseline/critic podľa experimentu. Strategický filter akcií je v hlavných
kvantových porovnaniach vypnutý.

## Čo znamená „kvantový“ v tomto projekte

Kvantové obvody sa v týchto experimentoch **simulujú klasicky**; projekt
nepoužíva kvantový hardvér a výsledky nepreukazujú kvantovú výhodu. Prvý
hybridný experiment používa PennyLane `default.qubit`, zatiaľ čo optimalizovaná
verzia počíta statevector diferenciovateľnými PyTorch operáciami.

Kompaktný model používa 4 qubity a 2 vrstvy obvodu. Kvantový blok vytvára
lokálne Z očakávania a susedné ZZ korelácie. Ich použitie samo osebe nie je
dôkazom kvantového previazania. Varianty s naučiteľnými a zmrazenými
kvantovými parametrami pomáhajú oddeliť efekt reprezentácie od efektu
optimalizácie parametrov obvodu.

## Výsledok, ktorý je už v priečinku

Úspešný experiment `results/efficiency_seed42_fixed/` porovnáva tri modely
na seedoch 42, 43 a 44 pri rozpočtoch 500, 1 000, 2 000, 5 000 a 10 000
tréningových zápasov. Každý checkpoint sa greedy evaluuje v 1 000 zápasoch
proti zmrazenému súperovi `v2`. Súhrn v
`results/efficiency_analysis_01/final_summary.csv` uvádza pri 10 000 zápasoch:

| Model | Parametre policy (celkom / trénovateľné) | Priemerná úspešnosť proti v2 |
|---|---:|---:|
| medium | 1 825 / 1 825 | 51,93 % ± 1,38 p. b. |
| compact_quantum | 389 / 389 | 54,90 % ± 2,81 p. b. |
| compact_frozen | 389 / 365 | 55,67 % ± 3,59 p. b. |

Ide o výsledky konkrétneho experimentu a jeho simulovaného prostredia. Samy
osebe nepreukazujú všeobecnú prevahu kvantových modelov. Najmä podobný výsledok
zmrazeného a učeného kvantového bloku znamená, že z týchto dát nemožno
pripisovať úspech výhradne učeniu kvantových rotácií. Časy tréningu sú
zaznamenané v rovnakom súhrne; simulované kvantové varianty sú podstatne
pomalšie než `medium`.

## Ako sa projekt vyvíjal

Nasledujúca postupnosť je **logická rekonštrukcia podľa aktuálnych skriptov,
konfigurácií a výsledkov**, nie presná commitová história. V repozitári zatiaľ
nie sú vytvorené Git commity.

1. **Hra a prvá RL policy.** `dice_rl.py` obsahuje základnú reprezentáciu hry,
   legálnych akcií a policy. `dice_rl_v2.py` je neskoršia verzia herného
   prostredia a reprezentácie, ktorú používajú hlavné porovnávacie experimenty.
2. **Stabilnejšie tréningové baseline.** `dice_rl_critic.py` pridáva actor–critic
   tréning. `dice_rl_scalar_control.py` skúma jednoduchý skalárny baseline;
   `compare_training_seeds.py` porovnáva tréningové prístupy a seedy.
3. **Klasické architektúry a ich limity.** `compare_policy_sizes.py` porovnáva
   rôzne veľkosti policy. Nadväzujúce skripty skúmajú rovnakú hĺbku,
   zúžené/bottleneck a taperované siete. Diagnostické skripty analyzujú
   konkrétne rozhodnutia a správanie pri bankovaní.
4. **Prvý kvantovo-klasický baseline.** `compare_quantum_classical.py`
   porovnáva stredne veľký klasický model, klasický blok a hybridný blok.
   `hybrid_no_ent` umožňuje porovnať obvod bez entangling CNOT vrstvy.
5. **Rýchlejšia simulácia a kompaktné modely.** `quantum_block_fast.py`
   implementuje presnú diferenciovateľnú simuláciu statevectorom v PyTorch.
   `compare_compact_quantum.py` skúma kompaktný kvantový blok, variant bez
   entanglementu a klasickú kontrolu s podobnou veľkosťou.
6. **Sample efficiency a ablačné testy.** `compare_sample_efficiency.py`
   ukladá checkpointy pri viacerých tréningových rozpočtoch a zahŕňa model so
   zmrazenými kvantovými parametrami. `analyze_sample_efficiency.py` vytvára
   tabuľky a grafy z CSV; `diagnose_efficiency_checkpoints.py` vykonáva
   dodatočnú kontrolu checkpointov.
7. **Analýza správania a prezentácia.** Analytické skripty vytvárajú
   porovnania architektúr, rozhodnutí a sample-efficiency. `generate_presentation.py`
   generuje `Quantum_RL_Dice_Presentation.pptx`.

## Súbory v koreňovom priečinku

### Herné prostredie a základné tréningy

- `dice_rl.py` — skoršia implementácia hry a základného učenia policy.
- `dice_rl_v2.py` — aktuálna herná logika použitá vo väčšine experimentov:
  stav, skórovanie, legálne akcie a súperi.
- `dice_rl_critic.py` — policy, value network/critic, spracovanie zápasu a
  actor–critic tréning.
- `dice_rl_scalar_control.py` — alternatívny tréning so skalárnym baseline.
- `dice_policy*.pt` — uložené checkpointy rôznych skorších policy; názov
  checkpointu identifikuje variant, ktorý treba pri načítaní zodpovedajúco
  interpretovať.

### Experimenty a modely

- `compare_policy_sizes.py` — porovnanie klasických veľkostí policy.
- `compare_training_seeds.py` — porovnanie tréningových nastavení/seedov.
- `compare_equal_depth.py` — porovnanie modelov pri vyrovnanej hĺbke.
- `compare_bottleneck.py` — porovnanie klasických bottleneck variantov.
- `compare_tapered.py` — porovnanie taperovanej klasickej siete s existujúcimi
  variantmi.
- `compare_quantum_classical.py` — hlavný experiment hybridnej
  kvantovo-klasickej policy.
- `quantum_block_fast.py` — PyTorch statevector simulátor kvantového bloku.
- `compare_compact_quantum.py` — compact quantum, no-entanglement a klasická
  kontrolná architektúra.
- `compare_sample_efficiency.py` — tréning s viacerými checkpoint rozpočtami
  a variantom so zmrazeným quantum blokom.
- `compare_sampling_vs_greedy.py` — porovnanie sampled a greedy výberu akcií.

### Analýzy a diagnostika

- `analyze_policy_sizes.py` — grafy architektúr a rozbor zvolených akcií.
- `analyze_sample_efficiency.py` — súhrny a grafy sample-efficiency.
- `diagnose_bottleneck.py` — diagnostika bottleneck policy.
- `diagnose_narrow.py` — diagnostika úzkej policy.
- `diagnose_tapered_decisions.py` — diagnostika rozhodnutí taperovaných/
  kvantových policy.
- `diagnose_efficiency_checkpoints.py` — diagnostika uložených checkpointov
  sample-efficiency.
- `generate_presentation.py` — zdrojový skript prezentácie.
- `Quantum_RL_Dice_Presentation.pptx` — vygenerovaná prezentácia.

## Výsledkové priečinky

Väčšina behov zapisuje konfiguráciu (`config.json`), CSV, checkpointy `.pt`
a podľa skriptu aj grafy. Názvy priečinkov identifikujú jednotlivé porovnania:

- `results/smoke_test/` — krátke smoke testy tréningových variantov.
- `results/main_10k/` — hlavné behy critic a scalar-control baseline.
- `results/policy_sizes_smoke/`, `results/policy_sizes_10k/` — klasické policy
  rôznych veľkostí.
- `results/equal_depth_smoke/`, `results/equal_depth_10k/` — porovnanie
  klasických sietí rovnakej hĺbky.
- `results/bottleneck4_10k/`, `results/tapered_10k/`,
  `results/tapered_decisions/` — bottleneck, taperované modely a rozhodnutia.
- `results/policy_analysis/` — grafy architektúr a prípadová analýza rozhodnutia.
- `results/quantum_main_01/`, `results/no_ent_10k/` — pôvodné kvantové
  porovnanie a variant bez entangling vrstvy.
- `results/quantum_fast_pilot_01/`, `results/quantum_fast_pilot_v2/` — pilot
  a následné behy s rýchlym simulátorom.
- `results/compact_pilot_01/`, `results/compact_main_01/` — pilot a hlavné
  porovnanie kompaktných architektúr.
- `results/efficiency_main_01/` — neúplný/prerušený prvý pokus o
  sample-efficiency; nepoužívať ako úplný súbor výsledkov.
- `results/efficiency_seed42_fixed/` — dokončený sample-efficiency dataset
  pre tri seedy a všetky päť tréningových rozpočtov.
- `results/efficiency_analysis_01/` — tabuľky a grafy vytvorené z vyššie
  uvedeného dokončeného datasetu.
- `results/efficiency_diagnostics_01/` — dodatočné diagnostické výsledky.
- `results/` obsahuje tiež textové výstupy `sampling_vs_greedy.txt` a
  `test_sampling_vs_greedy.txt`.

Checkpointy, CSV a grafy sú výstupy behov; nie sú všetky potrebné na spustenie
zdrojového kódu. Pred opakovaním experimentu zvoľ nový prázdny `--output-dir`,
pretože viacero skriptov odmietne zapisovať do existujúceho neprázdneho
priečinka.

## Spúšťanie

Príkazy sa spúšťajú z koreňového priečinka projektu, napríklad
`C:\diplomka`. Hlavný experiment vyžaduje referenčný checkpoint
`dice_policy_v2.pt`; predvolené nastavenia sú uložené v každom experimente do
`config.json`.

Príklad nového behu hybridného porovnania:

```powershell
python compare_quantum_classical.py `
  --models medium classical hybrid `
  --seeds 42 43 44 `
  --episodes 10000 `
  --eval-matches 1000 `
  --output-dir results\quantum_run_new
```

Príklad analýzy existujúceho sample-efficiency CSV:

```powershell
python analyze_sample_efficiency.py `
  --csv results\efficiency_seed42_fixed\efficiency.csv `
  --output-dir results\efficiency_analysis_new
```

Výstupný priečinok analýzy musí byť nový alebo prázdny. V koreňovom priečinku
nie je samostatný dependency manifest; verzie Pythonu, PyTorch a PennyLane
použité v konkrétnych quantum behoch sú zaznamenané v ich `config.json`.
