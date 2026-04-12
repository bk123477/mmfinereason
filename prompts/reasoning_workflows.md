# Multimodal Reasoning Workflows

Ten structured reasoning workflows for multimodal visual question answering.

Each workflow is optimized for a specific problem type. Select the single most appropriate
workflow based on the question and the content of the existing reasoning.

These workflows describe what to think about and in what order — they are invisible scaffolding,
not output templates. Never write section names, bracket headers, or phase labels in the reasoning.

---

## Workflow 1 — Structured Decomposition (SD)

**Best for:** Complex multi-step problems that require systematic, phase-by-phase analysis.
Math word problems, multi-condition geometry, science problems with multiple unknowns.
Use this as the **default** structure when no other workflow fits more naturally.

**Flow:**
**Problem Understanding** — State precisely what the question is asking. Identify the problem type, what information is given, what constraints exist, and what needs to be found.
**Visual Analysis** — Systematically examine the image. Identify all visible elements, labels, measurements, spatial arrangements, and any information embedded in diagrams or figures.
**Information Extraction** — List all relevant values, formulas, constraints, and relationships extracted from both the question text and the image. Retrieve any applicable domain knowledge, definitions, or theorems needed to solve the problem.
**Solution Strategy** — Decide the overall approach — which formulas, theorems, or reasoning steps to apply, and in what order. Justify why this strategy is appropriate for the problem.
**Step-by-Step Execution** — Carry out the solution plan with explicit calculations or logical deductions at each step. Show all intermediate work without skipping steps.
**Verification & Answer** — Cross-check the result against the question and image for consistency. Confirm the answer satisfies all given conditions, then state it clearly and concisely.

---

## Workflow 2 — Observe-Hypothesize-Test (OHT)

**Best for:** Ambiguous visual scenes, visual puzzles, pattern recognition problems,
or any case where the answer is not immediately apparent and hypotheses must be tested.

**Flow:**
**First Impression** — Note the immediate visual impression — what the image appears to show at a glance, and what the question seems to be probing.
**Detailed Observation** — Carefully examine each component, region, or element. Record anything informative, unusual, or relevant.
**Hypothesis Formation** — Based on observations, propose one or more candidate answers or interpretations of the image.
**Evidence Testing** — Search the image for specific evidence that confirms or refutes each hypothesis. Reason about which is best supported.
**Conclusion** — Commit to the best-supported hypothesis. State the final answer.

---

## Workflow 3 — Question-Guided Search (QGS)

**Best for:** Questions that ask for a specific piece of information (a value, label, object, count, or location)
that can be directly found or derived from the image with targeted lookup.

**Flow:**
**Question Decomposition** — Break down the question — what specific information is requested? What type of answer is expected (number, label, option)?
**Search Target Definition** — Identify exactly which visual feature, region, or value needs to be found to answer the question.
**Visual Localization** — Scan the image to find the target. Describe where it is and what it shows.
**Direct Reasoning** — Apply any necessary computation or inference to convert the located information into the final answer.
**Answer** — State the final answer concisely.

---

## Workflow 4 — Constraint Satisfaction (CS)

**Best for:** Logic puzzles, grid puzzles, Zebra-style puzzles, elimination-based multiple choice,
or any problem where multiple rules or conditions must simultaneously hold.

**Flow:**
**Constraint Inventory** — List every rule, condition, or requirement given in the question text.
**Visual Constraint Reading** — Extract additional constraints visible in the image — measurements, labels, diagrams, spatial arrangements.
**Option Space** — Enumerate all candidate answers or solution states before filtering.
**Constraint Propagation** — Apply each constraint to eliminate impossible options or narrow the solution space. Show reasoning step by step.
**Solution** — State the unique remaining solution that satisfies all constraints.

---

## Workflow 5 — Schema Recognition & Application (SRA)

**Best for:** Images that follow a well-known domain format: bar/line/pie charts, scatter plots,
tables, graphs, circuit diagrams, flowcharts, Venn diagrams, maps, chemical structures.

**Flow:**
**Schema Identification** — Identify the type of diagram or visual format. Name the domain (e.g., "bar chart comparing annual sales", "electric circuit with resistors in series").
**Structural Reading** — Read the structural components — axes, legend, labels, nodes, connections, scales, units.
**Data Extraction** — Extract the specific values, categories, or relationships from the diagram that are relevant to the question.
**Schema-Based Inference** — Apply domain knowledge appropriate to this diagram type to reason about the extracted data.
**Answer Derivation** — Derive the final answer from the inference and state it clearly.

---

## Workflow 6 — Comparative Analysis (CompA)

**Best for:** Problems that ask to compare, rank, order, or distinguish between multiple
distinct elements, objects, options, or datasets visible in the image.

**Flow:**
**Element Identification** — Enumerate all elements, objects, or categories to be compared. Confirm they are all visible/defined.
**Individual Characterization** — Analyze each element separately — record its key property, value, or attribute relevant to the comparison.
**Systematic Comparison** — Compare elements along the relevant dimension. Work through all pairs or rank all candidates explicitly.
**Key Distinction** — Identify the critical difference or distinguishing feature that determines the answer.
**Answer** — State which element satisfies the question's criterion.

---

## Workflow 7 — Spatial-Geometric Reasoning (SGR)

**Best for:** Geometry problems, coordinate geometry, spatial relationship questions,
measurement tasks, angle/area/volume computations, 2D/3D transformation problems.

**Flow:**
**Spatial Context** — Establish the coordinate system, orientation, or reference frame. Note any given measurements, scale, or constraints.
**Geometric Element Identification** — Identify all relevant geometric entities — points, lines, angles, shapes, dimensions, coordinates, and their labels.
**Relationship Mapping** — Determine geometric relationships — parallel, perpendicular, congruent, similar, inscribed, tangent, etc. State which theorems apply.
**Computation** — Apply the relevant geometric theorems, formulas, or trigonometry. Show each calculation step.
**Answer** — State the numerical result with appropriate units, or the geometric conclusion.

---

## Workflow 8 — Causal-Process Analysis (CPA)

**Best for:** Science questions about processes, cycles, mechanisms, or sequences.
Diagrams showing cause-and-effect chains, biological cycles, physical transformations, chemical reactions.

**Flow:**
**Process Overview** — Identify the overall process or system depicted. State what is happening at a high level and what domain it belongs to.
**Component Identification** — Identify each component, stage, entity, or participant in the process and describe its role.
**Causal Chain Tracing** — Trace the sequence of causes and effects, inputs and outputs, or before-and-after transitions step by step.
**Mechanism Explanation** — Explain the underlying principle, law, or mechanism that drives each step in the process.
**Answer to Question** — Apply the causal understanding to directly answer the specific question asked.

---

## Workflow 9 — Evidence Accumulation (EA)

**Best for:** Complex visual scenes or dense images where the answer requires synthesizing
multiple scattered clues before any single piece alone is sufficient to conclude.

**Flow:**
**Scene Assessment** — Establish the overall context — what type of scene is this, what is the setting, and what general information is available.
**Evidence Scan — Pass 1** — Collect high-level visual facts — objects present, approximate positions, dominant features, overall structure.
**Evidence Scan — Pass 2** — Collect fine-grained details — text labels, numerical values, specific markings, subtle features missed in the first pass.
**Evidence Integration** — Combine all collected evidence into a coherent interpretation relevant to the question. Explain how different clues relate.
**Reasoned Conclusion** — Draw the conclusion supported by the full body of evidence. State the final answer.

---

## Workflow 10 — Self-Checking Iterative Solution (SCIS)

**Best for:** High-stakes computations, tricky multi-step problems where a mistake is easy to make,
or problems where the initial intuitive answer should be verified before committing.

**Flow:**
**Problem Framing** — State precisely what is given, what is unknown, and what type of answer is expected. Identify potential pitfalls or common errors for this problem type.
**Initial Solution** — Solve the problem using the most natural approach. Work through it completely.
**Verification Check** — Re-examine the image and question. Check whether the initial solution is internally consistent, has correct units, and matches all visible evidence.
**Error Detection & Correction** — If any inconsistency is found, diagnose the error and rework the affected steps. If no error is found, explicitly confirm correctness.
**Final Answer** — State the verified final answer with full confidence.

---

## Workflow Selection Guide

| Problem Type | Recommended Workflow |
|---|---|
| Multi-step math / algebra | SD or SGR or SCIS |
| Geometry / spatial measurement | SGR |
| Visual puzzle / pattern recognition | OHT or CS |
| Logic puzzle / elimination | CS |
| Chart / graph / table / diagram | SRA |
| Comparing multiple elements | CompA |
| Direct lookup of fact/value | QGS |
| Science process / mechanism | CPA |
| Dense scene with multiple clues | EA |
| Tricky problem needing verification | SCIS |
| Complex multi-condition problem | SD or CS |
