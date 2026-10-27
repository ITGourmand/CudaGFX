# CudaGFX — Dossier de spécification pour IA (v2)

> **Mode d'emploi**
> 1. Donne ce fichier en entier à l'IA (pièce jointe ou collé).
> 2. Colle le « Prompt d'amorçage » de la section 1 comme premier message.
> 3. Avance phase par phase (section 11). Ne demande jamais « tout le projet d'un coup ».
>
> **Ce qui change en v2** : le document intègre les vérifications faites sur `bgfx_compute.sh` et sur `shaderc 1.19.149`, et la stratégie passe à **zéro dépendance externe** (plus de Clang, front-end maison, linkage statique, shaders embarqués).
>
> Légende : ✅ = vérifié sur les fichiers réels. ⚠️ = **à vérifier** dans les sources installées, jamais de mémoire.

---

## Table des matières

1. Prompt d'amorçage
2. Contexte
3. Objectifs et non-objectifs
4. Contraintes non négociables
5. Architecture (dont politique de dépendances)
6. Interface en ligne de commande
7. Langage d'entrée : sous-ensemble CUDA supporté
8. Correspondances CUDA → `.sc` (dialecte shaderc de bgfx)
9. Vérifications bgfx / shaderc (statut)
10. Code host généré et runtime `cudagfx_rt.h`
11. Plan de travail par phases et critères d'acceptation
12. Diagnostics
13. Tests
14. Instructions de travail pour l'IA
15. Erreurs à ne pas faire
16. Exemple de bout en bout
17. Références

---

## 1. Prompt d'amorçage

À coller tel quel comme premier message :

````text
Tu es un ingénieur senior en compilateurs et en programmation GPU/graphique
(front-ends C/C++, bgfx, shaderc, HLSL/GLSL, CUDA). Tu vas implémenter "CudaGFX"
en suivant le document joint (CudaGFX_spec_prompt.md) à la lettre.

Mission : écrire un transpileur source-à-source en C++17, SANS AUCUNE DEPENDANCE
EXTERNE (ni LLVM/Clang, ni Boost), linkable statiquement, qui lit du code avec la
syntaxe CUDA (kernels __global__, <<<grid, block>>>, threadIdx, __shared__,
__syncthreads...) et produit :
  (a) un fichier C++ standard (compilable par g++, clang++ et MSVC) pour la partie host,
      avec les numéros de ligne du .cu préservés ;
  (b) un ou plusieurs shaders compute .sc dans le dialecte de bgfx, compilables par shaderc ;
  (c) un header de stubs de lancement et de shaders embarqués (via shaderc --bin2c).

Règles absolues :
- Lis tout le document avant de répondre. Respecte les sections 4 (contraintes)
  et 15 (erreurs à ne pas faire).
- Les points marqués "vérifié" dans la section 9 sont acquis : ne les redevine pas.
  Tout point marqué "A VERIFIER" doit être contrôlé dans les sources de bgfx
  présentes dans le projet, jamais deviné de mémoire. Si tu ne peux pas vérifier,
  dis-le explicitement et propose une hypothèse clairement étiquetée.
- Ne génère jamais un shader faux en silence : en cas de doute, émets un
  diagnostic d'erreur avec fichier:ligne:colonne.
- Travaille phase par phase (section 11). Ne saute pas de phase.
- Fichiers complets, pas de "..." ni de "le reste est identique".
- N'ajoute aucune dépendance sans me le demander.

Format de ta PREMIÈRE réponse uniquement (n'écris aucun code dans celle-ci) :
  1. Ta reformulation du projet en 10 lignes maximum.
  2. Les hypothèses que tu fais et les points A VERIFIER que tu dois contrôler.
  3. Les risques techniques que tu vois (notamment pour le front-end maison).
  4. Ton plan détaillé pour la Phase 0.
Puis attends mon "go".
````

---

## 2. Contexte

### 2.1 Le besoin

L'utilisateur développe en C++ avec **bgfx** (projet « GDEngine », sous Linux, bgfx via `bgfx.cmake` dans `.deps/`). Il veut écrire du code de calcul GPU avec la **vraie syntaxe CUDA** (`__global__`, `<<<>>>`, `threadIdx.x`, `__shared__`…), tout en restant **portable** : le même source doit fonctionner sur tous les backends de bgfx via **shaderc**, sans GPU NVIDIA ni SDK NVIDIA.

Contrainte directrice de la v2 : **le moins de dépendances possible et le plus statique possible**, aussi bien pour l'outil que pour le runtime généré.

### 2.2 L'idée retenue : un transpileur, pas un compilateur

```
mon_code.cu ─► [lexer + mini-préprocesseur + parseur device maison] ─► IR typée
                                      ├─► emitter shader ─► cs_xxx.sc ─► shaderc (--bin2c) ─► tableaux C embarqués
                                      └─► emitter host   ─► main_gen.cpp + kernels.h ─► g++ / clang++ / MSVC
```

- Le fichier `.cu` d'entrée **n'est pas** compilable par g++ (à cause de `<<<>>>`). Seul CudaGFX le lit.
- La sortie host est du **C++ standard** : c'est ce qui garantit la compatibilité gcc/g++/clang/MSVC.
- Le **code host de l'utilisateur n'est jamais parsé en profondeur** : il est traité comme du texte opaque, sauf quelques régions repérées par le lexer (inclusions, définitions de kernels, lancements).
- Seul le **code device** (kernels et fonctions `__device__`) est parsé complètement, car il doit être traduit.

### 2.3 Ce qu'est « le HLSL de bgfx » (✅ vérifié sur `bgfx_compute.sh`)

Ce n'est pas du HLSL brut. Les shaders compute sont des fichiers `.sc` qui font `#include <bgfx_compute.sh>`. Faits établis :

- ✅ `NUM_THREADS(x, y, z)` : en GLSL, la macro s'étend en `layout(local_size_x…) in;` **avec son point-virgule** ; ailleurs en `[numthreads(x, y, z)]`. On l'écrit donc **sans** point-virgule.
- ✅ `BUFFER_RO(nom, type, slot)`, `BUFFER_RW(nom, type, slot)`, `BUFFER_WO(nom, type, slot)`. **Il n'existe pas de `BUFFER_WR`.** L'expansion dépend du backend :
  - GLSL : bloc `layout(std430, binding=slot) <accès> buffer <nom>Buffer { type nom[]; }`
  - Metal, SPIR-V, WGSL : `StructuredBuffer<T>` / `RWStructuredBuffer<T>`
  - HLSL, DXIL et autres : **buffers typés** `Buffer<T>` / `RWBuffer<T>`
- ✅ `SHARED` = `shared` (GLSL) ou `groupshared` (autres).
- ✅ `barrier()` : natif en GLSL, macro vers `GroupMemoryBarrierWithGroupSync()` ailleurs. Existent aussi `memoryBarrierShared()`, `memoryBarrierBuffer()`, etc.
- ✅ Atomiques : `atomicAdd/And/Max/Min/Or/Xor(mem, data)`. Sur GLSL ils retournent l'ancienne valeur ; ailleurs ils s'étendent en `InterlockedXxx(mem, data)` qui **ne retourne rien**. La forme portable avec valeur de retour est `atomicFetchAndAdd(mem, data, original)` (et `…And`, `…Max`, `…Min`, `…Or`, `…Xor`, `…Exchange`, `atomicFetchCompareExchange(mem, compare, data, original)`).
- ✅ Images : `IMAGE2D_RO/WO/RW`, `UIMAGE2D_*`, versions `ARRAY` et `3D`, avec formats (`r32ui`, `r32f`, `rg32ui`, `rgba32ui`, `rgba32f`, …), `imageLoad`, `imageStore`, `imageSize`, `imageAtomicAdd` (uint `r32ui`).
- ✅ Le fichier déclare automatiquement `uniform vec4 bgfx_indirectArgBase;` : les identifiants `bgfx_*` sont réservés.
- ⚠️ `gl_LocalInvocationID`, `gl_WorkGroupID`, `gl_GlobalInvocationID`, `gl_LocalInvocationIndex` **ne sont pas définis dans ce fichier**. Ils sont fournis ailleurs (`bgfx_shader.sh` ou shaderc lui-même) : voir V2.

### 2.4 Décisions déjà prises

| Sujet | Décision |
|---|---|
| Nature de l'outil | Transpileur source-à-source |
| Nom | **CudaGFX**, binaire `cudagfx`, namespace C++ `cudagfx`, préfixe de diagnostics `CGFX` |
| Front-end | **Maison, sans dépendance** : lexer, mini-préprocesseur, parseur du sous-ensemble device, analyse sémantique. Pas de Clang/LLVM |
| Linkage de l'outil | Statique autant que la plateforme le permet |
| Sortie shader | `.sc` dialecte bgfx, imprimé **depuis l'AST typé** |
| Embarquement des shaders | `shaderc --bin2c` : tableaux C dans un header, aucune lecture de fichier à l'exécution |
| Sortie host | C++17 standard, numéros de ligne du `.cu` préservés |
| Runtime | `cudagfx_rt.h`, header-only, bgfx + bibliothèque standard uniquement |
| Environnement connu | Linux, `shaderc` 1.19.149 construit par `bgfx.cmake` (`.deps/bgfx.cmake/build/cmake/bgfx/shaderc`), CMake |
| Langue | Commentaires en français, identifiants en anglais |

### 2.5 Questions ouvertes (valeur par défaut si l'utilisateur ne tranche pas)

| Question | Défaut |
|---|---|
| Front-end maison ou Clang lié statiquement ? | Maison (voir 5.2 pour le compromis). Clang n'est pas exclu comme **outil de test externe optionnel** |
| Taille de bloc dynamique côté host | Non supportée : une variante de shader par taille de bloc constante |
| Lecture de buffers (`cudaMemcpy` DeviceToHost) | Via texture, avec un jeu **fermé** de petits shaders embarqués (10.7) |
| Streams CUDA | Ignorés (un seul flux ordonné) |
| Embarquement | `--bin2c` par défaut, chargement de fichiers = option de développement |

---

## 3. Objectifs et non-objectifs

### Objectifs

- Écrire `saxpy<<<g, 256>>>(...)` et que ça s'exécute via bgfx sur n'importe quel backend.
- Outil **autonome** : un exécutable unique, compilé depuis les sources par le CMake du projet (comme `shaderc` dans `bgfx.cmake`), sans toolchain externe en dehors d'un compilateur C++17.
- Garder le code host de l'utilisateur **intact** (même numérotation de lignes, même texte hors régions modifiées).
- Produire des shaders lisibles, corrects, compilables par shaderc.
- Refuser **bruyamment** ce qui n'est pas supportable, avec fichier, ligne et colonne.
- Être utilisable comme étape de build (`add_custom_command`) comme `moc` ou `protoc`.

### Non-objectifs

- Ne **pas** être un compilateur CUDA complet ni viser la compatibilité totale avec nvcc.
- Ne **pas** parser le C++ host arbitraire. Le host reste opaque.
- Ne **pas** supporter les warp intrinsics, cooperative groups, dynamic parallelism, `double`, structs dans les buffers, templates dans les kernels (en v1).
- Ne **pas** écrire de backend HLSL/SPIR-V/MSL : on génère du `.sc`, shaderc fait le reste.
- Ne **pas** viser les performances de CUDA natif.

---

## 4. Contraintes non négociables

- **C1.** Le code C++ généré compile avec **g++, clang++ et MSVC** en C++17, sans `__attribute__`, sans `#pragma` spécifique à un compilateur, sans extension GNU. (Des macros vides comme `#define __host__` dans `cudagfx_rt.h` sont autorisées.)
- **C2.** **Zéro dépendance externe pour l'outil** : C++17 et bibliothèque standard uniquement. Ni LLVM/Clang, ni Boost, ni Python. Aucun processus externe obligatoire (sauf `shaderc` quand `--shaderc` est demandé). Linkage statique autant que la plateforme le permet.
- **C3.** Les `<<<>>>` sont reconnus par le **lexer** (tokens dédiés), jamais par regex ni par macro. Le host n'est jamais parsé en profondeur.
- **C4.** Aucun shader faux en silence : tout cas non supporté produit un diagnostic `CGFXxxxx` avec `fichier:ligne:colonne` et un code de sortie non nul.
- **C5.** Le `.sc` produit compile avec shaderc pour au moins : `windows` + `s_5_0`, `linux` + `spirv`, `linux` + `430`, `osx` + `metal`. Optionnels : `s_6_x` (DXIL), `wgsl`, ESSL `310_es` et plus.
- **C6.** Le layout des paramètres de kernel (packing des uniforms, numéros de slots) est défini **à un seul endroit** (`cudagfx_abi.h`) inclus à la fois par l'outil et par le runtime.
- **C7.** Les numéros de ligne du code host sont préservés : toute région supprimée est remplacée par le même nombre de retours à la ligne. Les erreurs de compilation pointent ainsi vers le bon endroit du `.cu`.
- **C8.** Le comportement asynchrone de bgfx est documenté et encapsulé par le runtime, jamais caché sous une fausse promesse de synchronisme.
- **C9.** Aucune hypothèse sur bgfx ou shaderc tirée de la mémoire : vérifier dans les sources (section 9).
- **C10.** Chaque phase se termine par du code compilable et des tests qui passent.
- **C11.** Les shaders sont **embarqués statiquement** (`--bin2c`). Charger des fichiers est une option de développement, jamais le comportement par défaut.
- **C12.** Le runtime n'exige ni exceptions ni RTTI, ne fait aucune entrée/sortie fichier, et ne dépend que de bgfx et de la bibliothèque standard.

---

## 5. Architecture

### 5.1 Pipeline interne

1. **Lexer** : tokens C++ complets (chaînes, raw strings, commentaires, littéraux numériques), lignes de préprocesseur, et tokens `<<<` / `>>>` reconnus en contexte de lancement.
2. **Mini-préprocesseur** (voir 7.4) : uniquement pour l'**analyse**. Le texte host n'est jamais remplacé par sa version préprocessée.
3. **Repérage des régions** : définitions `__global__`, `__device__`, `__host__ __device__`, `__constant__`, lancements `nom<<<…>>>(…)`.
4. **Parseur du sous-ensemble device** → AST.
5. **Analyse sémantique** : résolution de noms, types, conversions C implicites, évaluation de constantes, fermeture des appels de fonctions, classification des paramètres, validations (section 12).
6. **Emitter shader** : AST typé → `.sc`.
7. **Emitter host** : réécriture textuelle **par régions**, préservant les lignes.
8. **Driver shaderc** (optionnel) : compile chaque `.sc` pour chaque cible, avec `--bin2c`, puis génère le header de shaders embarqués.

### 5.2 Pourquoi un front-end maison, et ce que ça coûte

| | Front-end maison (défaut) | Clang libTooling lié statiquement |
|---|---|---|
| Dépendances | Aucune | LLVM/Clang à compiler ou installer (libs statiques pas toujours fournies par les distributions) |
| Binaire | Petit, compilé en quelques secondes avec le projet | Lourd |
| Stabilité | Aucune API tierce | L'API de Clang change à chaque version majeure |
| Effort initial | **Plus élevé** : lexer, mini-préprocesseur, parseur, sémantique | Plus faible |
| Fidélité C++ | Seulement le sous-ensemble device (déjà imposé par les limites de HLSL/GLSL) | Totale |
| Risque | Écarts sémantiques à maîtriser (conversions entières, etc.) | Faible |

Le choix est cohérent avec la contrainte « minimum de dépendances » car le sous-ensemble device est de toute façon proche du « C avec structs » : un parseur dédié est réaliste. Le risque d'écart sémantique se gère par une suite de tests golden importante et, **si Clang est disponible sur la machine de développement**, par des tests différentiels optionnels (compiler le kernel en C++ via un adaptateur CPU et comparer). Clang ne doit jamais être requis pour construire ou utiliser CudaGFX.

### 5.3 Arborescence du dépôt

```
cudagfx/
├─ CMakeLists.txt
├─ README.md
├─ docs/verifications.md        # résultat des vérifications V1..V17 (section 9)
├─ tools/cudagfx/               # l'outil (C++17, stdlib seule)
│  ├─ main.cpp                  # CLI
│  ├─ Lexer.{h,cpp}
│  ├─ Preprocessor.{h,cpp}      # sous-ensemble (7.4)
│  ├─ Regions.{h,cpp}           # repérage kernels / device / lancements
│  ├─ Parser.{h,cpp}            # sous-ensemble device -> AST
│  ├─ Sema.{h,cpp}              # types, conversions C, constantes, validations
│  ├─ ShaderEmitter.{h,cpp}     # AST typé -> .sc
│  ├─ HostEmitter.{h,cpp}       # réécriture par régions + kernels.h
│  ├─ ShadercDriver.{h,cpp}     # shaderc + --bin2c
│  └─ Diagnostics.{h,cpp}
├─ include/cudagfx/
│  ├─ cudagfx_abi.h             # packing uniforms + slots (partagé outil/runtime)
│  ├─ cudagfx_rt.h              # runtime bgfx (header-only)
│  └─ cudagfx_compat.h          # optionnel : pour que l'IDE comprenne __global__, threadIdx...
├─ runtime_shaders/             # shaders internes écrits à la main (readback), compilés au build
├─ cmake/CudaGFX.cmake          # fonction cudagfx_add_kernels(...)
├─ examples/
│  ├─ handwritten/              # phase 0 : tout écrit à la main
│  ├─ saxpy/
│  ├─ reduce/
│  └─ transpose/
└─ tests/
   ├─ golden/                   # entrée .cu -> sortie attendue .sc / .cpp / .h
   ├─ negative/                 # un test par diagnostic
   ├─ shaderc/                  # compilation des .sc sur chaque cible
   └─ runtime/                  # exécution bgfx vs référence CPU
```

### 5.4 Politique de dépendances et de linkage

**Outil `cudagfx`**
- C++17 et bibliothèque standard uniquement. Pas de réseau, pas de script, pas de génération de code au moment de sa compilation.
- Linux : `-static` si possible, sinon `-static-libstdc++ -static-libgcc`. MSVC : `/MT`. macOS : le statique complet n'existe pas, lier statiquement la bibliothèque C++ quand c'est possible.
- Construit depuis les sources par le CMake du projet consommateur, à côté de `shaderc`.

**Runtime généré**
- Header-only, dépend uniquement des headers publics de bgfx et de la bibliothèque standard.
- Compilable avec `-fno-exceptions -fno-rtti`.
- Aucun fichier lu à l'exécution : shaders embarqués.

**Chaîne de build**
- `shaderc` vient de `bgfx.cmake` (même version que bgfx). L'intégration CMake localise la cible `shaderc`, elle n'en suppose pas le chemin.

**Outils externes facultatifs (jamais requis)** : `clang`/`g++` pour des tests différentiels.

---

## 6. Interface en ligne de commande

```bash
# cas simple (erreur claire si le .cu contient plusieurs kernels et un seul fichier de sortie)
cudagfx saxpy.cu -o saxpy_gen.cpp --shader-out cs_saxpy.sc

# plusieurs kernels : % est remplacé par le nom du kernel
cudagfx saxpy.cu -o saxpy_gen.cpp --shader-out "cs_%.sc"

# pipeline complet : shaderc + embarquement statique pour plusieurs cibles
cudagfx saxpy.cu -o saxpy_gen.cpp --shader-dir gen/ \
        --shaderc ./shaderc -i bgfx/src \
        --target linux:spirv --target linux:430 --target windows:s_5_0 --target osx:metal \
        --embed

# fonctions simples de debug
cudagfx saxpy.cu --dump-ir
```

| Option | Rôle |
|---|---|
| `<input>` | `.cu`, ou `.cpp` contenant de la syntaxe CUDA |
| `-o <fichier>` | Sortie host C++ (numéros de ligne préservés) |
| `--kernels-header <fichier>` | Header généré (stubs + shaders embarqués). Défaut : `<stem de -o>_kernels.h` |
| `--shader-out <chemin>` | Fichier `.sc` ou motif avec `%` |
| `--shader-dir <dossier>` | Dossier de sortie des shaders (nom : `cs_<kernel>[_<taille>].sc`) |
| `--shaderc <chemin>` | Active la compilation des `.sc` avec cet exécutable |
| `-i <dossier>` | Dossier d'include pour shaderc (celui qui contient `bgfx_compute.sh`, c.-à-d. `bgfx/src`). Répétable |
| `--target <plateforme>:<profil>` | Répétable. Plateformes ✅ : `android`, `asm.js`, `ios`, `linux`, `orbis`, `osx`, `windows`. Profils ✅ : `s_4_0`, `s_5_0`, `s_6_0`…`s_6_9`, `metal` (+ variantes), `spirv` (+ variantes), `120`…`440`, `100_es`…`320_es`, `pssl`, `wgsl`. Compute en GLSL : 430 ou plus, en ESSL : 310_es ou plus (⚠️ à confirmer) |
| `--embed` | Passe `--bin2c <nom>` à shaderc et génère le header de shaders embarqués |
| `--depends` | Fichier de dépendances style makefile (utilise `--depends` de shaderc et celui de l'outil) |
| `--no-line-preserve` | Désactive la préservation des lignes (non recommandé) |
| `--dump-ir` | Écrit l'IR en JSON (debug) |
| `--runtime-include <chemin>` | Include du runtime dans la sortie (défaut : `cudagfx_rt.h`) |
| `-D<nom>[=<val>]`, `-I<dossier>` | Passés au mini-préprocesseur |

Codes de sortie : `0` succès, `1` erreurs dans le source utilisateur (diagnostics `CGFX`), `2` erreur d'usage de la CLI, `3` échec de shaderc, `4` erreur interne.

---

## 7. Langage d'entrée : sous-ensemble CUDA supporté

### 7.1 Supporté en v1

| Domaine | Détail |
|---|---|
| Kernels | `__global__ void`, `__launch_bounds__(N)` (taille de bloc 1D) |
| Lancement | `k<<<grid, block>>>(args)` avec `grid`/`block` entiers ou `dim3`. `block` doit être évaluable à la compilation |
| Index | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`.x .y .z`) |
| Mémoire partagée | `__shared__ T nom[CONST]` (taille statique) |
| Synchronisation | `__syncthreads()` |
| Types scalaires | `float`, `int`, `unsigned`/`unsigned int`, `bool` |
| Types vectoriels | `float2/3/4`, `int2/3/4`, `uint2/3/4`, `make_floatN(...)`. **Pas** en élément de buffer pour les 3 composantes (section 8.4) |
| Fonctions device | `__device__` et `__host__ __device__`, appelées par les kernels, sans récursion |
| Math | `sinf cosf tanf sqrtf rsqrtf fabsf fminf fmaxf powf expf logf floorf ceilf fmodf`, `min`/`max`/`abs` sur entiers |
| Contrôle | `if/else`, `for`, `while`, `do/while`, `switch`, `break/continue/return` |
| Données locales | Variables locales, tableaux locaux de taille constante, structs POD **locales** (pas dans les buffers) |
| Constantes | `const`/`constexpr` globales scalaires, `#define` |
| Paramètres | scalaires, vecteurs (hors tableaux d'éléments 3 composantes), pointeurs vers scalaires ou vecteurs 2/4 composantes (`T*` ou `const T*`), utilisés **uniquement par indexation** `p[i]` |

API host supportée en v1 : `cudaMalloc`, `cudaFree`, `cudaMemcpy` (HostToDevice, DeviceToDevice, DeviceToHost en phase 5), `cudaMemset`, `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaGetErrorString`, `cudaError_t`, `cudaMemcpyKind`, `dim3`. Ces fonctions sont définies inline dans `cudagfx_rt.h` : **le code host qui les appelle n'est pas réécrit**.

### 7.2 Prévu en v2

Atomiques `int`/`uint` (`atomicAdd`, `atomicMin`, `atomicMax`, `atomicExch`, `atomicCAS`, …) avec les contraintes de 8.5, `__constant__` (vers uniforms), structs dans les buffers (limité aux backends qui les supportent, voir 8.4), blocs 2D/3D avancés, templates de kernels instanciés, `auto`, swizzles, streams (ignorés), `__ldg` (ignoré).

### 7.3 Refusé avec un diagnostic

- Pointeurs avec arithmétique libre, alias de pointeurs, pointeur de pointeur, pointeur dans une struct, pointeur passé à une fonction `__device__`.
- `double`, entiers 64 bits, `size_t`, `long` (v1).
- Éléments de buffer à 3 composantes (`float3*`, `int3*`, `uint3*`), structs dans les buffers (v1).
- Warp intrinsics : `__shfl*`, `__ballot*`, `__syncwarp`, `__any`, `__all`.
- `extern __shared__` (taille dynamique).
- `printf`, `assert`, `new`/`delete`/`malloc` dans un kernel.
- Récursion, fonctions virtuelles, exceptions, RTTI, lambdas dans un kernel.
- Dynamic parallelism (kernel qui en lance un autre), cooperative groups.
- Lancement avec taille de bloc non constante, ou dépassant les limites du backend.
- Lancement `<<<>>>` situé dans le corps d'une macro.

### 7.4 Traitement du code host et préprocesseur intégré

**Le host est opaque.** L'outil n'a besoin de repérer que :
1. les lignes `#include` (pour remplacer `<cuda_runtime.h>`),
2. les définitions de kernels et de fonctions device (à retirer du host),
3. les lancements `nom<<<cfg>>>(args)` (à remplacer),
4. les constantes de fichier utilisées dans les configurations de lancement ou dans les kernels.

Tout le reste (templates, classes, STL, lambdas, etc. dans le code host) passe tel quel. Un lancement à l'intérieur d'un template host ou d'une fonction membre fonctionne, puisque le stub est une fonction C++ ordinaire.

**Mini-préprocesseur** (utilisé pour l'analyse seulement) :

| Supporté | Détail |
|---|---|
| `#define` / `#undef` | objet et fonction, sans `##` ni `#` en v1 (diagnostic sinon) |
| `#if` `#ifdef` `#ifndef` `#elif` `#else` `#endif` | expressions entières, `defined()` |
| `#include "fichier.h"` | suivi (chemins `-I`) pour collecter code device et constantes |
| `#include <...>` | ignoré (host-only). Un symbole device défini uniquement là → diagnostic CGFX0010 |
| `#pragma once` | pris en compte, autres `#pragma` ignorés |
| Macros prédéfinies | `__CUDAGFX__` = 1. `__CUDA_ARCH__` **n'est pas défini** en v1 |

Les régions situées dans une branche inactive d'un `#if` sont ignorées pour l'analyse **mais conservées** dans le texte host.

**Détection des `<<<` / `>>>`** : après un identifiant (éventuellement qualifié), la séquence `<<<` ouvre une configuration de lancement. La fermeture est le premier `>>>` à profondeur 0 de parenthèses, crochets et accolades. Les arguments de templates avec `>>` imbriqués doivent être entre parenthèses (diagnostic sinon).

---

## 8. Correspondances CUDA → `.sc`

### 8.1 Table de correspondance

| CUDA | Sortie `.sc` |
|---|---|
| `threadIdx.x` | `gl_LocalInvocationID.x` (avec cast explicite si utilisé comme `int`) |
| `blockIdx.x` | `gl_WorkGroupID.x` |
| `blockDim.x` | littéral issu de la variante de bloc (`256`) |
| `gridDim.x` | uniform rempli par le runtime au lancement |
| bloc `256` (site de lancement) / `__launch_bounds__(256)` | `NUM_THREADS(256, 1, 1)` **sans point-virgule** ✅ |
| `__shared__ float t[256]` | `SHARED float t[256];` ✅ |
| `__syncthreads()` | `barrier();` ✅ |
| `const T* p` | `BUFFER_RO(p, T, slot);` ✅ |
| `T* p` | `BUFFER_RW(p, T, slot);` ✅ |
| `T* p` en écriture seule | `BUFFER_WO(p, T, slot);` (identique à RW hors GLSL) ✅ |
| `int n`, `float a`, `float3 v` | packés dans `uniform vec4 u_<kernel>_args[N]` |
| `float2/3/4` | `vec2/3/4` |
| `int2/3/4` | `ivec2/3/4` |
| `unsigned`, `uint2/3/4` | `uint`, `uvec2/3/4` |
| `make_float3(a, b, c)` | `vec3(a, b, c)` |
| `sqrtf`, `fabsf`, `fminf`, `fmaxf` | `sqrt`, `abs`, `min`, `max` |
| `rsqrtf(x)` | `inversesqrt(x)` (⚠️ à vérifier dans `bgfx_shader.sh`) |
| `fmodf(a, b)` | `mod(a, b)` : sémantique de signe différente, avertissement |
| `atomicAdd(&p[i], v)` | voir 8.5 |
| `k<<<g, b>>>(args)` | appel du stub généré (section 10) |

### 8.2 Règles de packing des paramètres (définies dans `cudagfx_abi.h`)

- Chaque scalaire occupe une lane 32 bits d'un `vec4`.
- `int`, `uint`, `bool` sont transmis par **bit-cast** (`memcpy` côté host, `floatBitsToInt` / `floatBitsToUint` côté shader, ⚠️ voir V4). Bool : `0` ou `1`.
- `float2` commence sur une lane paire. `float3`/`float4` commencent sur la lane 0 d'un slot. Aucun paramètre ne chevauche deux `vec4`.
- **Slots de buffers** : numérotés dans l'ordre de déclaration des paramètres pointeurs, en partant de 0, et **uniques** sur l'ensemble RO + RW + WO. Raison ✅ : hors GLSL, un buffer lecture est un registre `t#` et un buffer écriture un registre `u#`, mais bgfx lie par numéro de stage, donc deux buffers ne doivent jamais partager un numéro.
- Si un même pointeur device est passé deux fois au même lancement, le runtime détecte l'alias et signale une erreur (CGFXR003).

### 8.3 Typage strict : l'émission se fait toujours depuis l'AST typé

Le dialecte `.sc` est fortement typé et ses conversions implicites diffèrent de celles du C. L'analyse sémantique **implémente les règles du C** et l'émetteur écrit **un cast explicite pour chaque conversion implicite** :

- promotions entières (`bool` → `int`) ;
- conversions arithmétiques usuelles : si un opérande est `float` → `float`, sinon si un opérande est `uint` → `uint`, sinon `int` (pas de 64 bits en v1) ;
- `gl_WorkGroupID` et `gl_LocalInvocationID` sont **non signés** : `int i = int(gl_WorkGroupID.x) * 256 + int(gl_LocalInvocationID.x);`
- `%` et `/` sur entiers **signés** : le comportement pour opérandes négatifs peut différer selon le backend, avertissement CGFX0017.
- littéraux flottants : écrits sous une forme acceptée par tous les backends (⚠️ à tester en phase 0).

Il n'y a **pas** de mode « copier le texte du corps » : le texte source n'est jamais recopié tel quel dans le shader.

### 8.4 Éléments de buffer autorisés ✅

Les buffers sont **typés** sur plusieurs backends (`Buffer<T>`/`RWBuffer<T>` hors Metal/SPIR-V/WGSL) et en `std430` sur GLSL. Conséquences :

| Type d'élément | Statut en v1 | Raison |
|---|---|---|
| `float`, `int`, `uint` | ✅ autorisé | scalaire 32 bits |
| `float2`, `int2`, `uint2` | ✅ autorisé | 2 × 32 bits |
| `float4`, `int4`, `uint4` | ✅ autorisé | 4 × 32 bits |
| `float3`, `int3`, `uint3` | ❌ refusé (CGFX0012) | pas de format typé 3 composantes portable, et `vec3` en `std430` a un pas de 16 octets : layout host et device divergeraient |
| `struct` | ❌ refusé en v1 (CGFX0002) | `StructuredBuffer` seulement sur Metal/SPIR-V/WGSL, pas sur les buffers typés |
| `double`, 64 bits, `bool` | ❌ refusé | non portables |

⚠️ À confirmer en V5 : les flags de création de buffers compute de bgfx (`BGFX_BUFFER_COMPUTE_FORMAT_*` / `BGFX_BUFFER_COMPUTE_TYPE_*`) doivent refléter ce jeu fermé de 9 types.

### 8.5 Atomiques ✅ (v2)

- Disponibles uniquement sur `int` et `uint`, dans un buffer `RW` ou en mémoire `SHARED`. **Pas de `float`**, pas de 64 bits.
- Hors GLSL, `atomicAdd(mem, v)` ne retourne rien. Pour un `atomicAdd` CUDA dont on **utilise** la valeur de retour, il faut émettre au niveau **instruction** :
  - `int old = atomicAdd(&a[i], 1);` → `int old; atomicFetchAndAdd(a[i], 1, old);`
  - `x = atomicAdd(&a[i], 1);` → `atomicFetchAndAdd(a[i], 1, x);`
  - `atomicAdd(&a[i], 1);` (valeur ignorée) → `atomicAdd(a[i], 1);`
- Un `atomicXxx` dont le résultat est utilisé **au milieu d'une expression** → diagnostic CGFX0015 (le déplacement d'effets de bord n'est pas fait en v2).
- `&a[i]` est la seule forme de pointeur acceptée, et uniquement en argument d'un atomique : elle devient `a[i]`.

### 8.6 Noms réservés et collisions ✅

- Réservés : préfixe `bgfx_`, ainsi que `gl_`.
- En GLSL, `BUFFER_*` génère un bloc nommé `<nom>Buffer` : un identifiant utilisateur `xBuffer` pourrait entrer en collision avec le paramètre `x`. L'émetteur renomme défensivement ses identifiants (préfixe interne) ou refuse (CGFX0016).

---

## 9. Vérifications bgfx / shaderc (statut)

L'IA consigne chaque résultat dans `docs/verifications.md` (point, fichier consulté, conclusion). Les lignes ✅ sont acquises ; les lignes ⚠️ sont à faire en phase 0.

| # | Statut | Point | Résultat ou où regarder |
|---|---|---|---|
| V1 | ✅ | Macros compute | `NUM_THREADS`, `BUFFER_RO/RW/WO`, `SHARED`, `barrier()` : voir 2.3. Pas de `BUFFER_WR` |
| V2 | ⚠️ | Variables intégrées (`gl_LocalInvocationID`, `gl_WorkGroupID`, `gl_GlobalInvocationID`, index linéaire) et leurs types | Absentes de `bgfx_compute.sh`. Regarder `bgfx/src/bgfx_shader.sh` et le code de `bgfx/tools/shaderc` |
| V3 | ✅ | Types d'éléments des `BUFFER_*` | Buffers typés hors Metal/SPIR-V/WGSL, `std430` en GLSL : voir 8.4 |
| V4 | ⚠️ | Fonctions de bit-cast (`floatBitsToInt`, `floatBitsToUint`) sur tous les backends | `bgfx_shader.sh`. Indice : `floatBitsToUint` est utilisé dans une macro commune de `bgfx_compute.sh`. Repli si absent : refuser les entiers > 2^24 en uniform ou passer par un buffer |
| V5 | ⚠️ | Création des buffers compute côté host, flags `BGFX_BUFFER_COMPUTE_*`, correspondance avec le jeu de 9 types de 8.4 | `bgfx/include/bgfx/bgfx.h`, `defines.h`, exemples compute |
| V6 | ⚠️ | Signatures de `bgfx::setBuffer`, `bgfx::setUniform` (tableaux de `vec4`), `bgfx::dispatch`, `bgfx::createComputeProgram` | `bgfx.h` |
| V7 | ⚠️ | Ordre d'exécution de plusieurs dispatches dans une vue (`ViewMode::Sequential` ?) | `bgfx.h`, doc |
| V8 | ⚠️ | Lecture de buffer : existence ou non d'un readback direct ; sinon `bgfx::blit` + `BGFX_TEXTURE_READ_BACK` + `bgfx::readTexture`, délai en frames | `bgfx.h`, doc, exemples |
| V9 | ✅ | CLI de shaderc 1.19.149 | Voir ci-dessous |
| V10 | ✅ | Atomiques | Voir 8.5 |
| V11 | ⚠️ | Directives `#line` dans les `.sc` (non nécessaires pour le host, utiles pour les erreurs de shaderc) | test réel |
| V12 | ⚠️ | Limites par backend : nombre de threads par groupe, mémoire partagée, taille maximale de texture | doc de chaque API graphique, `bgfx::getCaps()` |
| V13 | ⚠️ | Création et nommage des `uniform vec4 nom[N]` (`createUniform(name, UniformType::Vec4, N)`) | `bgfx.h`, exemples |
| V14 | ⚠️ | `bgfx/embedded_shader.h` : convention de nommage des tableaux par backend (`_glsl`, `_spv`, `_dx11`, `_mtl`, `_essl`…) et sélection selon `bgfx::getRendererType()`. Faire correspondre les noms passés à `--bin2c` | `bgfx/include/bgfx/embedded_shader.h` |
| V15 | ⚠️ | Un shader compute demande-t-il `--varyingdef` ? (probablement non) | test avec shaderc en phase 0 |
| V16 | ⚠️ | Quelle valeur de `--platform` utiliser avec `--profile wgsl` | test avec shaderc |
| V17 | ⚠️ | Fonctions CMake de `bgfx.cmake` pour compiler les shaders (`bgfx_compile_shaders`, option `AS_HEADERS`…) : à réutiliser plutôt que réécrire | `bgfx.cmake/cmake/` |

### V9 — Ce que dit `shaderc -h` (version 1.19.149) ✅

- Usage : `shaderc -f <in> -o <out> --type <vertex|fragment|compute> --platform <platform> [-p <profile>]`.
- Plateformes : `android`, `asm.js`, `ios`, `linux`, `orbis`, `osx`, `windows`.
- Profils :
  - HLSL : `s_4_0`, `s_5_0` ; DXIL : `s_6_0` à `s_6_9`
  - Metal : `metal`, `metal10-10` … `metal31-14`
  - SPIR-V : `spirv`, `spirv10-10`, `spirv13-11`, `spirv14-11`, `spirv15-12`, `spirv16-13`
  - GLSL : `120`, `130`, `140`, `150`, `330`, `400`, `410`, `420`, `430`, `440` ; ESSL : `100_es`, `300_es`, `310_es`, `320_es`
  - `pssl`, `wgsl`
- **Il n'y a pas de profil `cs_5_0`** : le shader model compute D3D s'appelle `s_5_0` avec `--type compute`.
- Options utiles : `-i` (répétable), `--stdout`, **`--bin2c [nom]`** (génère un header C : c'est le mécanisme d'embarquement statique), **`--depends`** (fichier de dépendances style makefile), `--preprocess`, `--define`, `--raw`, `--verbose`, `--debug` (Vulkan/DirectX/Metal), `-O 0..3`, `--Werror`, `--disasm` (DirectX).

Commandes à valider en phase 0 :

```bash
./shaderc -f cs_saxpy.sc -o cs_saxpy.dx11.bin --type compute --platform windows --profile s_5_0 -i <bgfx>/src
./shaderc -f cs_saxpy.sc -o cs_saxpy.spv.bin  --type compute --platform linux   --profile spirv  -i <bgfx>/src
./shaderc -f cs_saxpy.sc -o cs_saxpy.glsl.bin --type compute --platform linux   --profile 430    -i <bgfx>/src
./shaderc -f cs_saxpy.sc -o cs_saxpy.mtl.bin  --type compute --platform osx     --profile metal  -i <bgfx>/src

# embarquement statique (la sortie est un header C contenant le tableau)
./shaderc -f cs_saxpy.sc -o cs_saxpy_spv.h --type compute --platform linux --profile spirv -i <bgfx>/src --bin2c cs_saxpy_spv
```

---

## 10. Code host généré et runtime `cudagfx_rt.h`

### 10.1 Réécriture du code host (par régions, lignes préservées)

L'emitter host travaille sur le texte d'origine avec des remplacements **par région**. Modifications autorisées, et **seulement** celles-ci :

- La ligne `#include <cuda_runtime.h>` est remplacée **une ligne pour une ligne** par `#include "<stem>_kernels.h"`. Si elle n'existe pas, l'inclusion est ajoutée tout en haut, suivie de `#line 1 "<fichier>.cu"` pour restaurer la numérotation.
- Les définitions `__global__` et les fonctions `__device__`-seules sont remplacées par **autant de retours à la ligne** qu'elles occupaient (elles vivent dans le `.sc`). Les fonctions `__host__ __device__` restent dans la sortie host **et** sont émises dans le shader.
- Chaque `k<<<g, b>>>(args)` devient `cudagfx_gen::k::launch(g, b, args)` en conservant le même nombre de lignes.

Tout le reste du code utilisateur est préservé. Le header `<stem>_kernels.h` contient : l'inclusion de `cudagfx_rt.h`, les stubs, et les shaders embarqués.

### 10.2 Stub généré (forme de référence)

```cpp
namespace cudagfx_gen {
struct saxpy {
    static constexpr unsigned block[3] = {256, 1, 1};
    static void launch(dim3 grid, dim3 blk, float a, const float* x, float* y, int n) {
        cudagfx::Launch l(shaders::saxpy_256());          // table statique de binaires embarqués
        l.arg(a); l.arg(n);                                // packing défini dans cudagfx_abi.h
        l.template buffer<float>(0, x, cudagfx::Access::Read);       // type d'élément connu statiquement
        l.template buffer<float>(1, y, cudagfx::Access::ReadWrite);
        l.dispatch(grid, blk, block);                      // vérifie blk == block, puis bgfx::dispatch(...)
    }
};
} // namespace cudagfx_gen
```

- `shaders::saxpy_256()` renvoie une structure statique avec un pointeur et une taille par backend construit. Si le backend courant n'a pas été construit, le runtime signale une erreur nommant le profil manquant (CGFXR004).
- La forme exacte de l'API du runtime est laissée à l'IA, tant que les contraintes de la section 4 sont respectées.

### 10.3 Pointeurs device : garder `float*` tel quel

En CUDA, un pointeur device est opaque côté host. Le runtime fait pareil :

```cpp
inline cudaError_t cudaMalloc(void** p, size_t bytes) {
    *p = cudagfx::detail::alloc(bytes);   // faux pointeur unique, jamais déréférencé
    return cudaSuccess;                   // table interne : faux pointeur -> allocation
}
```

- Une table interne associe chaque plage `[ptr, ptr + taille)` à une allocation.
- `dx + offset` est résolu par recherche d'intervalle. Un offset non aligné sur la taille de l'élément produit une erreur claire.
- `cudaFree` libère l'allocation et le handle bgfx associé.

### 10.4 Buffers typés et création paresseuse (conséquence de 8.4)

`cudaMalloc` ne connaît pas le type d'élément, mais un buffer compute bgfx est **typé à sa création** (⚠️ V5). Le type n'est connu qu'au **premier lancement** qui lie ce buffer, car le stub le fournit statiquement. Donc :

1. `cudaMalloc` réserve une allocation « non typée » avec une copie hôte (*shadow*).
2. `cudaMemcpy` HostToDevice, `cudaMemset` et DeviceToHost avant tout lancement opèrent sur le shadow.
3. Au premier `buffer<T>(...)` le runtime crée le buffer bgfx typé, y envoie le contenu du shadow, puis libère le shadow.
4. Un lancement ultérieur qui lie la **même** allocation avec un autre type d'élément (par exemple un `float*` réinterprété en `uint*`) produit une erreur runtime (CGFXR002). Ce cas est rare et il est documenté comme limite connue.
5. La taille de l'allocation doit être un multiple de la taille de l'élément (sinon CGFXR001).

### 10.5 Cycle de vie et asynchronisme (C8)

- L'application initialise bgfx (`bgfx::init`) **avant** d'utiliser le runtime. CudaGFX fournit `cudagfx::init(...)` pour déclarer la vue à utiliser. Il ne crée pas de fenêtre.
- Les dispatches sont soumis à une vue dédiée. Ils s'exécutent réellement au `bgfx::frame()`.
- Le runtime accumule les dispatches puis force un `bgfx::frame()` à ces points : `cudaDeviceSynchronize`, `cudaMemcpy` DeviceToHost, ou appel explicite de `cudagfx::flush()`.
- Documenter ce comportement dans le README. Ne jamais prétendre qu'un lancement est synchrone.

### 10.6 Taille de bloc et variantes

- La taille de bloc est fixée à la compilation du shader (`NUM_THREADS`).
- Pour un kernel lancé avec plusieurs tailles de bloc constantes, l'outil génère une variante par taille (`cs_saxpy_256.sc`, `cs_saxpy_128.sc`) et le stub choisit la variante au lancement.
- Si le `block` d'un lancement n'est pas constant : **diagnostic d'erreur** (pas de repli silencieux).
- Si `__launch_bounds__` est présent et ne correspond à aucun site de lancement : avertissement.

### 10.7 Lecture des buffers (`cudaMemcpy` DeviceToHost) : conception minimale et statique

**Objectif** : aucune génération dynamique de shader, aucune lecture de fichier, aucune dépendance à un mécanisme propre à un backend.

**Fait de départ (⚠️ V8)** : à ma connaissance, bgfx permet de relire une **texture** (`BGFX_TEXTURE_READ_BACK`, `bgfx::readTexture`) mais pas directement un **buffer**. À confirmer sur la version installée ; si un readback direct de buffer existe, l'utiliser et supprimer le reste de cette section.

**Conception par défaut** :

1. **Jeu fermé de shaders.** Comme les types d'élément sont un jeu fermé de 9 types (8.4), le runtime embarque **au plus 9 petits shaders** de copie (un par type et nombre de composantes, **écrits à la main** dans `runtime_shaders/`). Ils sont compilés par shaderc au build du projet, pour les mêmes cibles que les kernels, puis embarqués avec `--bin2c`. Le jeu est fini et connu à la compilation. Les buffers typés ne pouvant pas être relus sous un autre type sur D3D, il faut une variante par type (réduction possible en phase 5 : lire chaque type, le convertir en `uint` par bit-cast, écrire dans une texture `r32ui` / `rg32ui` / `rgba32ui`).
2. **Chaîne de copie** : shader de copie → texture A (écriture compute, formats `r32ui`/`rg32ui`/`rgba32ui` ✅ disponibles dans `bgfx_compute.sh` via `UIMAGE2D_WO`) → `bgfx::blit` vers texture B (`READ_BACK`) → `bgfx::readTexture` → enchaînement de `bgfx::frame()` jusqu'à la frame retournée → copie vers le pointeur host.
3. **Découpage 2D** : l'indice linéaire est converti en coordonnées 2D avec une largeur fixe (≤ taille maximale de texture, V12). Au-delà de la capacité d'une texture, lecture en plusieurs morceaux.
4. **Textures de readback réutilisées** et libérées à `cudagfx::shutdown()`.
5. **Avant tout lancement** (buffer encore en shadow, 10.4) : `cudaMemcpy` DeviceToHost est une simple copie mémoire, sans GPU.

**Alternatives écartées par défaut** :
- Stocker toute la mémoire device sous forme de **textures 2D** à la place de buffers (readback direct, mais indexation `imageLoad/imageStore` à générer partout, atomiques limités, perte de performance).
- Un shader de copie **unique et générique, par octets** : impossible avec des buffers typés sur D3D.

### 10.8 Gestion d'erreurs du runtime

- Toutes les fonctions `cuda*` retournent un `cudaError_t` cohérent. `cudaGetLastError` / `cudaGetErrorString` fonctionnent.
- Codes d'erreur runtime : CGFXR001 taille non multiple de l'élément, CGFXR002 buffer relié avec deux types différents, CGFXR003 alias de pointeur dans un lancement, CGFXR004 shader absent pour le backend courant, CGFXR005 pointeur device inconnu, CGFXR006 taille de bloc incohérente avec la variante.
- Aucun `assert` silencieux en build release.

### 10.9 Intégration CMake (phase 4)

Fournir `cmake/CudaGFX.cmake` avec une fonction `cudagfx_add_kernels(TARGET <cible> SOURCES <fichiers.cu> TARGETS <plateforme:profil...>)` qui :

1. construit `cudagfx` (sous-projet, linkage statique) ;
2. localise `shaderc` via la cible de `bgfx.cmake` (⚠️ V17, pas de chemin codé en dur) ;
3. exécute `cudagfx` avec `add_custom_command` (dépendances via `--depends`) ;
4. ajoute les `.cpp` générés à la cible ;
5. compile les shaders internes du runtime (readback) de la même manière.

---

## 11. Plan de travail par phases et critères d'acceptation

### Phase 0 — Prototype entièrement manuel (sans outil) et vérifications

Livrables : `examples/handwritten/` avec `cs_saxpy.sc`, une mini-appli bgfx, un `cudagfx_rt.h` minimal écrit à la main, et `docs/verifications.md` (V2, V4 à V8, V11 à V17).

Critères d'acceptation :
- `cs_saxpy.sc` compile avec le `shaderc` du projet pour `linux:spirv`, `linux:430`, et pour toutes les autres cibles testables sur la machine.
- Un shader embarqué par `--bin2c` est chargé et exécuté sans fichier.
- Un test compare le résultat GPU à une référence CPU sur 1 M d'éléments.
- Les points ⚠️ de la section 9 listés ci-dessus ont une réponse écrite.

### Phase 1 — Outil MVP avec front-end maison

Périmètre : un seul kernel, paramètres scalaires et pointeurs (indexation seule), `blockDim` constant 1D. Lexer complet, mini-préprocesseur (`#define` objet, `#if/#ifdef`, `#include "…"`), parseur d'expressions et d'instructions du sous-ensemble, sémantique avec conversions C, émission shader depuis l'AST, réécriture host par régions.

Critères d'acceptation :
- `cudagfx saxpy.cu -o saxpy_gen.cpp --shader-out cs_saxpy.sc` produit les fichiers attendus.
- La sortie host compile avec g++, clang++ (et MSVC si disponible) sans warning en `-Wall -Wextra`, et **les numéros de ligne correspondent au `.cu`**.
- Tests golden sur `.sc`, host et header.
- Les erreurs de source produisent un diagnostic avec `fichier:ligne:colonne`.
- Binaire construit sans aucune bibliothèque externe, vérifié par `ldd` (Linux) : uniquement la libc, ou aucune dépendance en statique complet.

### Phase 2 — Kernels réalistes

Périmètre : plusieurs kernels, variantes de taille de bloc, fonctions `__device__` (fermeture des appels), `__shared__` + `__syncthreads`, blocs/grilles 2D et 3D, types vectoriels, structs POD locales, `#define` fonction.

Critères d'acceptation :
- Exemples `reduce` (réduction en mémoire partagée) et `transpose` (2D avec tuiles) compilent, s'exécutent et sont validés contre la CPU.
- Un kernel lancé avec deux tailles de bloc génère deux shaders.

### Phase 3 — Diagnostics complets et robustesse

Périmètre : tous les diagnostics de la section 12, évaluateur de constantes robuste, cas limites du lexer (raw strings, continuations de ligne, trigraphes ignorés), cas de casts `int`/`uint` mixtes.

Critères d'acceptation :
- Aucun cas de la liste « Refusé » (7.3) ne passe sans diagnostic.
- Un test négatif par code de diagnostic.
- Fuzz léger du lexer et du parseur (aucun crash, aucun blocage) sur des entrées aléatoires.

### Phase 4 — Driver shaderc, embarquement statique, CMake

Critères d'acceptation :
- `--shaderc … --target … --embed` produit les binaires et le header de shaders embarqués, inclus automatiquement par les stubs.
- Plusieurs cibles dans une même exécution.
- Échec de shaderc : code de sortie `3` et message reprenant la sortie de shaderc.
- `cudagfx_add_kernels(...)` fonctionne dans le projet GDEngine (CMake) avec le `shaderc` de `bgfx.cmake`.

### Phase 5 — Runtime complet

Périmètre : buffers typés à création paresseuse (10.4), readback (10.7), `cudaMemset`, `cudaMemcpy` DeviceToDevice, atomiques (8.5), détection d'alias, mesures de performance.

Critères d'acceptation :
- Cycle complet `cudaMalloc` → `cudaMemcpy` H2D → kernel → `cudaMemcpy` D2H → `cudaFree` validé sur au moins deux backends.
- Tests runtime automatisés comparant GPU et CPU.

### Phase 6 — Optionnelle

Backend CPU `--emit-cpu` (exécution des kernels par boucles sur la machine, pour tester sans GPU et faire des tests différentiels), templates de kernels instanciés, `__constant__`, structs dans les buffers sur les backends compatibles.

---

## 12. Diagnostics

Format : `fichier.cu:ligne:colonne: error: CGFXnnnn: message`, rendu comparable à celui d'un compilateur (extrait de la ligne et caret `^`).

| Code | Sens |
|---|---|
| CGFX0001 | Pointeur avec arithmétique / alias non supporté dans un kernel |
| CGFX0002 | Type non supporté (`double`, entier 64 bits, struct dans un buffer…) |
| CGFX0003 | Warp intrinsic non supporté |
| CGFX0004 | `extern __shared__` non supporté (taille statique requise) |
| CGFX0005 | Fonction non supportée dans un kernel (`printf`, `malloc`, récursion…) |
| CGFX0006 | Taille de bloc non constante au site de lancement |
| CGFX0007 | Taille de bloc incohérente avec `__launch_bounds__` (avertissement) |
| CGFX0008 | Taille de bloc ou de mémoire partagée au-delà des limites |
| CGFX0009 | Construction C++ non supportée dans un kernel (lambda, virtual, exception, template…) |
| CGFX0010 | Identifiant inconnu ou fonction `__device__` sans définition visible |
| CGFX0011 | Nom de kernel ambigu pour `--shader-out` (plusieurs kernels, un seul fichier) |
| CGFX0012 | Élément de buffer à 3 composantes non supporté (`float3*`…) |
| CGFX0013 | Lancement `<<<>>>` dans une macro, ou configuration de lancement mal formée |
| CGFX0014 | Directive de préprocesseur non supportée (`##`, `#` stringification, `#include_next`…) |
| CGFX0015 | Atomique dont le résultat est utilisé au milieu d'une expression, ou atomique sur un type non supporté (`float`) |
| CGFX0016 | Identifiant réservé (`bgfx_*`, `gl_*`) ou collision avec un nom généré |
| CGFX0017 | Avertissement : `%` ou `/` sur entiers signés, comportement potentiellement différent selon le backend |
| CGFX0100 | Erreur interne : à signaler (inclure le fichier source minimal) |

---

## 13. Tests

- **Golden** : entrée `.cu` → sortie `.sc` / `.cpp` / `_kernels.h` attendue, comparées par diff. Mise à jour via un flag explicite (`UPDATE_GOLDEN=1`), jamais automatique.
- **Lignes** : un test vérifie que chaque ligne du host généré hors régions modifiées est **identique** à la ligne du `.cu` au même numéro.
- **shaderc** : chaque `.sc` généré est compilé sur chaque cible disponible (au minimum `linux:spirv`, `linux:430`, plus `windows:s_5_0` et `osx:metal` si l'outil peut les générer sur la machine). Un échec est un échec de test.
- **Runtime** : chaque exemple s'exécute via bgfx et le résultat est comparé à une version CPU (tolérance flottante documentée).
- **Compilateurs** : la sortie host est compilée avec g++ et clang++ en `-std=c++17 -Wall -Wextra -Werror`, et `-fno-exceptions -fno-rtti` pour le runtime.
- **Négatifs** : un test par diagnostic de la section 12 (le source doit être refusé avec le bon code et la bonne position).
- **Dépendances** : un test vérifie que le binaire `cudagfx` n'a pas d'autre dépendance dynamique que la libc.
- **Différentiel (optionnel)** : si `clang` est présent, comparaison des résultats d'un adaptateur CPU compilé avec Clang à ceux du shader.
- Intégration continue via CTest.

---

## 14. Instructions de travail pour l'IA

1. **Lire** tout le document. Restituer la compréhension avant d'écrire du code (voir prompt d'amorçage).
2. **Vérifier** avant d'affirmer : pour chaque point ⚠️, consulter les sources installées (bgfx dans `.deps/bgfx.cmake/`, `shaderc` construit). Noter le résultat dans `docs/verifications.md`. Si impossible : le dire, étiqueter l'hypothèse.
3. **Une phase à la fois.** Ne pas anticiper les suivantes, sauf pour garder une architecture extensible.
4. **Fichiers complets.** Chaque fichier livré est entier et compilable. Pas de « … », pas de « idem ». Indiquer le chemin du fichier avant chaque bloc de code.
5. **Petits pas testables.** Chaque livraison inclut : le code, les commandes de build, les commandes de test, le résultat attendu.
6. **Décisions** : quand plusieurs options existent, en présenter deux au maximum avec un avis motivé, puis appliquer la valeur par défaut du document si l'utilisateur ne répond pas.
7. **Honnêteté** : signaler toute limite, tout contournement, tout test non exécuté. Ne jamais écrire « testé » sans avoir exécuté le test.
8. **Code de l'outil** : C++17, bibliothèque standard seule, pas d'exceptions (retours d'erreur explicites avec un type résultat), identifiants en anglais, commentaires en français, une responsabilité par fichier, pas de variable globale mutable.
9. **Diagnostics** : toute erreur utilisateur passe par le système de diagnostics (section 12), pas par des `std::cerr` éparpillés.
10. **Dépendances** : ne pas en ajouter sans le demander. L'outil n'a aucune dépendance. Le runtime n'a que bgfx.
11. **Format des réponses** : (a) résumé de ce qui est fait, (b) fichiers complets, (c) commandes de build/test, (d) points à vérifier restants, (e) proposition de la suite.
12. **Si le contexte devient trop long**, proposer un résumé d'état (phase en cours, fichiers existants, décisions prises, points ouverts) à coller dans une nouvelle conversation.

---

## 15. Erreurs à ne pas faire

### Conception et dépendances

- ❌ **Ajouter une dépendance** (LLVM/Clang, Boost, Python, bibliothèque de parsing…) à l'outil sans accord explicite.
- ❌ **Parser `<<<>>>` avec une regex ou une macro.** Un lexer dédié, point.
- ❌ **Essayer de parser le C++ host complet.** Il reste opaque : on ne touche que les régions repérées.
- ❌ **Écrire un compilateur complet** ou un backend HLSL/SPIR-V. On génère du `.sc`, shaderc s'occupe du reste.
- ❌ **Mélanger lexing, parsing, sémantique et émission** dans les mêmes fonctions. Passer par l'AST typé et l'IR.
- ❌ **Faire dépendre la sortie host de l'outil** : elle ne dépend que de bgfx et de `cudagfx_rt.h`.
- ❌ **Charger des shaders depuis le disque par défaut.** Embarquement statique via `--bin2c`.
- ❌ **Linker dynamiquement** quand le statique est possible.

### Front-end maison

- ❌ **Recopier le texte du corps du kernel dans le shader.** Toujours imprimer depuis l'AST typé.
- ❌ **Oublier les règles du C** : promotions entières, conversions arithmétiques usuelles, `int`/`uint` mixtes. Le shader doit avoir un cast explicite pour **chaque** conversion implicite.
- ❌ **Utiliser le préprocesseur pour transformer le texte host.** Il sert à l'analyse seulement.
- ❌ **Ignorer silencieusement une construction non reconnue** (« je saute ce qui ne se parse pas »). Toujours un diagnostic.
- ❌ **Calculer des positions de texte à la main** de façon fragile : chaque token garde fichier, ligne, colonne et offsets, et les réécritures utilisent ces offsets.
- ❌ **Casser la numérotation des lignes** du host. Une région supprimée est remplacée par le même nombre de retours à la ligne.
- ❌ **Se perdre dans les macros** : `##`, `#` et lancements dans une macro sont refusés en v1 avec un diagnostic clair.
- ❌ **Ne pas tester le lexer/parseur sur des entrées hostiles** (fichiers tronqués, chaînes non fermées, `<<<` isolés) : aucun crash, aucune boucle infinie.

### Génération du shader

- ❌ **Supposer que le shader est du HLSL brut.** C'est le dialecte `.sc` de bgfx.
- ❌ **Utiliser `BUFFER_WR`** (n'existe pas) : c'est `BUFFER_WO`.
- ❌ **Mettre un point-virgule après `NUM_THREADS(…)`** : la version GLSL en contient déjà un.
- ❌ **Utiliser `cs_5_0`** comme profil : le profil compute D3D est `s_5_0`.
- ❌ **Émettre `float3*` / `int3*` / `uint3*`** en élément de buffer, ou des structs dans les buffers en v1 (8.4).
- ❌ **Faire partager le même numéro de slot** à un buffer RO et à un buffer RW.
- ❌ **Écrire `int old = atomicAdd(&a[i], 1);` tel quel.** Hors GLSL, `atomicAdd` ne retourne rien : utiliser `atomicFetchAndAdd` au niveau instruction, et refuser les atomiques utilisés au milieu d'une expression.
- ❌ **Supposer des atomiques flottants**, 64 bits, ou sur `bool`.
- ❌ **Utiliser `blockDim` comme variable dynamique.** C'est un littéral issu de la variante de bloc.
- ❌ **Émettre des pointeurs** ou de l'arithmétique de pointeurs dans le `.sc`.
- ❌ **Inventer des macros ou fonctions bgfx.** Si ce n'est pas dans `bgfx_compute.sh` / `bgfx_shader.sh`, ça n'existe pas.
- ❌ **Ignorer les différences de comportement** de `fmodf`/`mod`, du `%` et de la division entière sur valeurs négatives, des conversions `float` → `int`.
- ❌ **Nommer un identifiant `bgfx_*`, `gl_*`**, ni laisser un nom utilisateur entrer en collision avec `<nom>Buffer` généré par la version GLSL de `BUFFER_*`.
- ❌ **Dépasser les limites** de taille de groupe ou de mémoire partagée des backends sans diagnostic.

### Uniforms et buffers

- ❌ **Définir le packing à deux endroits** (outil et runtime). Il vit uniquement dans `cudagfx_abi.h`.
- ❌ **Passer un `int` dans un uniform en le convertissant en `float`** sans le documenter : au-delà de 2^24 la valeur est faussée. Utiliser le bit-cast (après V4) ou refuser.
- ❌ **Faire chevaucher un paramètre sur deux `vec4`.**
- ❌ **Créer un buffer bgfx à `cudaMalloc`.** Il est typé à sa création : création paresseuse au premier lancement, avec copie shadow avant (10.4).
- ❌ **Lier le même buffer deux fois dans un lancement** sans détection d'alias, ni le relier sous un autre type sans erreur.

### Runtime, readback et bgfx

- ❌ **Prétendre que les lancements sont synchrones.** Les dispatches s'exécutent au `bgfx::frame()`.
- ❌ **Supposer un readback direct des buffers.** Vérifier V8 ; sinon passer par texture `READ_BACK` et attendre la bonne frame.
- ❌ **Générer ou compiler des shaders de readback à l'exécution.** Jeu fermé, écrit à la main, compilé et embarqué au build.
- ❌ **Ignorer les limites de taille de texture** pour le readback d'un grand buffer.
- ❌ **Oublier que l'état bgfx (`setBuffer`, `setUniform`) est par soumission** et réinitialisé après chaque `dispatch` (⚠️ à confirmer V6).
- ❌ **Supposer l'ordre d'exécution entre dispatches** sans avoir vérifié le mode de vue (V7).
- ❌ **Créer la fenêtre ou initialiser bgfx dans le runtime.** C'est le rôle de l'application.
- ❌ **Faire des E/S fichier, utiliser des exceptions ou du RTTI** dans le runtime.
- ❌ **Fuiter les handles bgfx** : tout ce qui est créé doit être détruit (`cudaFree`, `cudagfx::shutdown`).

### Qualité et processus

- ❌ **Générer un shader faux en silence.** Toujours un diagnostic.
- ❌ **Réécrire plus de code host que nécessaire.** Hors des modifications de 10.1, le code reste identique, lignes comprises.
- ❌ **Utiliser des extensions GNU/MSVC dans la sortie host** (`__attribute__`, `__declspec`…).
- ❌ **Livrer du code non compilable** ou des morceaux (`// ... reste du code`).
- ❌ **Écrire « testé »** sans avoir exécuté.
- ❌ **Sauter une phase.** Chaque phase repose sur la précédente.
- ❌ **Mettre à jour les fichiers golden automatiquement** pour « faire passer » les tests.
- ❌ **Cacher une limite connue.** Mieux vaut un diagnostic clair qu'un comportement surprenant.

---

## 16. Exemple de bout en bout

### 16.1 Entrée : `saxpy.cu`

```cpp
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

__global__ __launch_bounds__(256)
void saxpy(float a, const float* x, float* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main()
{
    const int n = 1 << 20;
    std::vector<float> hx(n, 1.0f), hy(n, 2.0f);

    float *dx = nullptr, *dy = nullptr;
    cudaMalloc((void**)&dx, n * sizeof(float));
    cudaMalloc((void**)&dy, n * sizeof(float));
    cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<(n + 255) / 256, 256>>>(2.0f, dx, dy, n);

    cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("y[0] = %f (attendu 4.0)\n", hy[0]);

    cudaFree(dx);
    cudaFree(dy);
    return 0;
}
```

### 16.2 Commande

```bash
cudagfx saxpy.cu -o saxpy_gen.cpp --shader-dir gen/ \
        --shaderc ./shaderc -i bgfx/src \
        --target linux:spirv --target linux:430 --embed
```

### 16.3 Sortie shader : `gen/cs_saxpy_256.sc` (forme attendue, ⚠️ à valider en phase 0)

```glsl
#include <bgfx_compute.sh>

// Généré par CudaGFX depuis saxpy.cu — ne pas modifier à la main
uniform vec4 u_saxpy_args[1];   // [0].x = a (float), [0].y = n (int, bit-cast)

BUFFER_RO(x, float, 0);
BUFFER_RW(y, float, 1);

NUM_THREADS(256, 1, 1)
void main()
{
    float a = u_saxpy_args[0].x;
    int   n = floatBitsToInt(u_saxpy_args[0].y);   // saxpy.cu:6 (signature)

    int i = int(gl_WorkGroupID.x) * 256 + int(gl_LocalInvocationID.x);   // saxpy.cu:8
    if (i < n) { y[i] = a * x[i] + y[i]; }                               // saxpy.cu:9
}
```

### 16.4 Sortie host : `saxpy_gen.cpp` (extraits attendus)

Les numéros de ligne sont identiques à ceux de `saxpy.cu` (le kernel, lignes 5 à 10, est remplacé par 6 lignes vides) :

```cpp
#include "saxpy_gen_kernels.h"          // ligne 1 : remplace #include <cuda_runtime.h>
#include <vector>                       // ligne 2
#include <cstdio>                       // ligne 3
                                        // ligne 4
                                        // lignes 5 à 10 : vides (ex-kernel __global__)
                                        //
                                        //
                                        //
                                        //
                                        //
                                        // ligne 11
int main()                              // ligne 12
{
    // ... lignes 14 à 21 inchangées ...
    cudagfx_gen::saxpy::launch((n + 255) / 256, 256, 2.0f, dx, dy, n);   // ligne 23
    // ... suite inchangée ...
}
```

### 16.5 Header généré : `saxpy_gen_kernels.h` (extraits)

```cpp
#pragma once
#include "cudagfx_rt.h"
// tableaux embarqués générés par shaderc --bin2c
#include "gen/cs_saxpy_256_spv.h"
#include "gen/cs_saxpy_256_glsl.h"

namespace cudagfx_gen {
namespace shaders { /* saxpy_256() : table statique par backend */ }
struct saxpy { /* stub de la section 10.2 */ };
}
```

### 16.6 Compilation finale

```bash
g++     -std=c++17 saxpy_gen.cpp -Iinclude -I<bgfx>/include -o saxpy -lbgfx -lbx -lbimg
# ou
clang++ -std=c++17 saxpy_gen.cpp -Iinclude -I<bgfx>/include -o saxpy -lbgfx -lbx -lbimg
```

---

## 17. Références

- bgfx : https://github.com/bkaradzic/bgfx et https://bkaradzic.github.io/bgfx/
- bgfx.cmake : https://github.com/bkaradzic/bgfx.cmake
- Fichiers de référence dans le dépôt bgfx : `src/bgfx_compute.sh`, `src/bgfx_shader.sh`, `include/bgfx/bgfx.h`, `include/bgfx/defines.h`, `include/bgfx/embedded_shader.h`, `tools/shaderc/`, dossier `examples/` (exemples compute)
- Guide de programmation CUDA : https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Slang (alternative à garder en tête si le besoin devient « langage compute C++-like multi-backend » sans syntaxe CUDA exacte) : https://shader-slang.com/
