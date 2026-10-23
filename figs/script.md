voice preference: female

This production-ready video script explains the SALEM (Self-improving Agentic Loops: LLM-Extracted features for Materials Property Prediction) project in 2 minutes and 50 seconds (roughly 410 words at a standard 145 words-per-minute narration pace). It integrates the project workflow diagram (feature_extraction_gp_pipeline_2x (1).jpg), all five experimental figures, audio direction, background music cues, and the full creator credits.   

Video Production Overview
- Title: SALEM: Self-Improving Agentic Loops for Materials Property Prediction   
- 
- Target Duration: ~2 minutes, 50 seconds (170 seconds)
- 
- Target Word Count: ~410 words
- 
- Overall Visual Style: Clean, modern academic motion graphics; smooth zoom-and-pan transitions across diagrams and data plots with subtle highlight callouts.
- 
Scene-by-Scene Script
Scene 1: Title & Project Creators (0:00 – 0:15)
- Visuals:
- 
  - Animated title card: SALEM: Self-improving Agentic Loops: LLM-Extracted features for Materials Property Prediction.   
  - 
  - Below the title, display a clean, structured grid of the project creators and their institutional affiliations:
  - 
    1. Sven P. Voigt — Maximal Intelligence
    2. 
    3. Mohd Zaki — Johns Hopkins University
    4. 
    5. Zhanzhao Li — Rice University
    6. 
    7. Defne Circi — Duke University
    8. 
    9. Xiaoquan Liu — Duke University
    10. 
    11. Eltjo Mante — Duke University
    12. 
    13. Michalis Mitsakis — WMG Sustainable Innovation
    14. 
- Audio Style: Warm, articulate, and engaging voiceover (VO); crisp studio quality with an inviting academic tone.
- 
- Background Music: Ambient electronic synth pad with a soft, forward-moving pulse (105 BPM). Starts at medium volume and ducks to 15% under the narration.
- 
- Narration (VO):
- 
- "Welcome to SALEM: Self-improving Agentic Loops for Materials Property Prediction. Created by researchers from Maximal Intelligence, Johns Hopkins University, Rice University, Duke University, and WMG Sustainable Innovation, SALEM introduces an automated framework for building reliable, self-improving scientific knowledge graphs."   
- 
Scene 2: The Motivation — Literature Scale & Diversity (0:15 – 0:45)
- Visuals:
- 
  - Display Fig_start.jpg full screen.   
  - 
  - 0:18: Zoom into Panel (a) (Abstracts mentioning perovskite surging to 4,732 papers in 2025) and Panel (b) (Subtopic phrases showing solar and halide research rising sharply after 2014).   
  - 
  - 0:30: Pan across to Panel (c) (arXiv categories dominated by Materials Science with 2,433 titles) and Panel (d) (28,340 title matches across PubMed, arXiv, and ChemRxiv).   
  - 
- Audio Style: Urgent, inquisitive, and analytical narration; subtle "whoosh" sound effects on panel transitions.
- 
- Background Music: Pulsing minimal marimba and light percussive ticks join the synth pad to convey rapid growth and data scale.
- 
- Narration (VO):
- 
- "Research on perovskite materials has exploded over the past two decades, surpassing 4,700 annual papers in 2025 and totaling over 28,000 title matches across PubMed, arXiv, and ChemRxiv. Spanning diverse subtopics from halide solar cells to photocatalysis, this massive literature makes manual data curation impossible. While AI agents can extract synthesis and experimental features at scale, how do we verify their accuracy when curated ground truth doesn't exist? SALEM solves this by using a single reliable reference property—curated peak photoluminescence—to quantitatively score and refine LLM-extracted features."   
- 
Scene 3: The End-to-End SALEM Workflow (0:45 – 1:35)
- Visuals:
- 
  - Transition to the workflow diagram feature_extraction_gp_pipeline_2x (1).jpg.   
  - 
  - 0:48 (Step 1): Highlight Box 1: Query & Corpus Retrieval. Show queries $Q$ retrieving relevant records from the HybriD3 corpus using Reciprocal Rank Fusion of BM25 and dense embeddings.   
  - 
  - 1:00 (Step 2): Pan right to Box 2: Feature Extraction ($D \times E = \text{Features}$). Highlight the three Extractor Operators in the legend: $f$ (Direct Extraction from structured datasets), $g$ (Domain Language Model: MatSciBERT + process variables), and $h$ (Generative QA Pipeline: LLM question-answering + embeddings).   
  - 
  - 1:15 (Step 3): Pan down to Box 3: Gaussian Process (GP). Highlight the partitioned target vector $y$ (peak photoluminescence) and input matrix $X$.   
  - 
  - 1:23 (Step 4): Pan left to Box 4: Loss Evaluation & Backpropagation. Animate the 5-fold cross-validation arrow ($\mathcal{E} = \sum_{i=1}^5 \mathcal{E}_i$) and the dashed feedback loops $\frac{\partial \mathcal{E}}{\partial q_i}$ ("Update Queries") and $\frac{\partial \mathcal{E}}{\partial E_j}$ ("Update Extractors").   
  - 
- Audio Style: Clear, methodical, and instructional pacing. Subtle digital chimes highlight each of the four pipeline stages.
- 
- Background Music: Rhythmic, interlocking synth arpeggios that evoke a closed-loop machine learning pipeline.
- 
- Narration (VO):
- 
- "Here is how the SALEM workflow operates end-to-end. First, in Query and Corpus Retrieval, hybrid search retrieves relevant perovskite records from the HybriD3 database. Second, in Feature Extraction, SALEM constructs a feature matrix by applying three operator types across the documents: direct structured extraction, MatSciBERT domain embeddings, and an agentic generative QA pipeline that asks targeted synthesis and experimental questions. Third, these extracted features serve as inputs $X$ to a Gaussian Process conditioned on the target property $y$—the photoluminescence peak. Fourth, using 5-fold cross-validation, SALEM computes finite-difference error gradients with respect to each question and query. If a question adds noise, the agent prunes or rewrites it; if a query lowers prediction error, the agent specializes around it—creating a self-improving feedback loop."   
- 
Scene 4: Experimental Results — Pruning & Complementarity (1:35 – 2:10)
- Visuals:
- 
  - 1:35: Display fig1.png (GP error vs. number of extracted questions). Highlight the sharp error drop at the first question (+ syn: metal, dropping from 98.5 to 86.9 nm on questions alone, and 84.4 to 76.8 nm with formula and conditions), followed by the upward creep as more questions are added.   
  - 
  - 1:52: Transition to fig2.jpg (Pairwise CV MAE and Redundancy/Complementarity heatmaps). Highlight the blue complementary cells (formula + syn: metal at 78.2 nm and formula + exp: sample form at 78.5 nm) versus red redundant/conflicting cells (formula + syn: halide at +4.3 nm and exp: excitation + syn: solute at +8.8 nm).   
  - 
- Audio Style: Insightful, data-driven delivery with vocal emphasis on key takeaways ("fewer, sharper questions win").
- 
- Background Music: Smooth, focused mid-tempo beat with warm bass notes supporting the analytical reveal.
- 
- Narration (VO):
- 
- "Testing on 209 HybriD3 records reveals why agentic pruning is essential. In Figure 1, adding the single best question—metal—cuts prediction error from 84.4 to 76.8 nanometers. However, blindly accumulating all ten questions degrades performance back to 85.9 nanometers. Fewer, sharper questions win. Figure 2's error landscape explains why: features like sample form and metal strongly complement the chemical formula, whereas halide is redundant with formula due to composition leakage, and solute conflicts with excitation."   
- 
Scene 5: Query Specialization & Future Outlook (2:10 – 2:50)
- Visuals:
- 
  - 2:10: Split-screen or sequential reveal of fig0.jpg and fig3.png.   
  - 
  - In fig0.jpg, highlight the error reductions for process queries: spin coat (-30%), laser excitation (-38%), and slow cooling (-52%, reaching 41.0 nm MAE) compared to all records (-17%).   
  - 
  - 2:25: In fig3.png (Leave-one-feature-out heatmap), highlight the dark blue cells showing critical unique features for specific queries, such as formula (-17.2 nm) and exp: instrument (-5.3 nm) under laser excitation, or se_max (-11.2 nm) under bromide.   
  - 
  - 2:38: Fade out into a closing visual of the Materials Tetrahedron (connecting Processing, Structure, Composition, and Properties) alongside the SALEM logo and team credits.   
  - 
- Audio Style: Inspiring, forward-looking conclusion that leaves a lasting impression.
- 
- Background Music: Swells slightly with uplifting harmonic chords, resolving cleanly on a final resonant note at 2:50.
- 
- Narration (VO):
- 
- "Furthermore, Figure 0 and Figure 3 demonstrate that agentically chosen queries dramatically lower prediction error and specialize the questions. While the global model reduces baseline error by 17 percent, local Gaussian Processes on process-specific queries like spin coating and slow cooling cut error by 30 to 52 percent. Looking ahead, SALEM provides a scalable blueprint for materials informatics—identifying optimal feature vectors to predict missing material properties across literature databases, and helping complete the materials tetrahedron."   