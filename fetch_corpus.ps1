# fetch_corpus.ps1 — Download real training data from Wikipedia + Project Gutenberg
# Usage: .\fetch_corpus.ps1 [-OutFile corpus.txt] [-ArticleCount 500] [-GutenbergBooks 20]
param(
    [string]$OutFile      = "wiki_corpus.txt",
    [int]$ArticleCount    = 500,
    [int]$GutenbergBooks  = 10,
    [int]$DelayMs         = 150,
    [switch]$SkipGutenberg
)

$ErrorActionPreference = "Continue"

# ── topic list (broad coverage = diverse training signal) ──────────────────
$Topics = @(
    "Mathematics","Physics","Chemistry","Biology","Computer_science",
    "Artificial_intelligence","Machine_learning","Neural_network","Deep_learning",
    "Natural_language_processing","Statistics","Probability","Calculus","Algebra",
    "Geometry","Number_theory","Information_theory","Graph_theory","Algorithm",
    "Data_structure","Operating_system","Compiler","Database","Cryptography",
    "Quantum_computing","Robotics","Neuroscience","Cognitive_science","Psychology",
    "Philosophy","Logic","Ethics","Economics","History","Geography","Astronomy",
    "Cosmology","Evolution","Genetics","Ecology","Climate","Thermodynamics",
    "Electromagnetism","Optics","Relativity","Mechanics","Fluid_dynamics",
    "Topology","Abstract_algebra","Linear_algebra","Differential_equation",
    "Fourier_analysis","Signal_processing","Control_theory","Game_theory",
    "Formal_language","Automaton","Computational_complexity_theory","NP-completeness",
    "Sorting_algorithm","Search_algorithm","Dynamic_programming","Greedy_algorithm",
    "Recursion","Functional_programming","Object-oriented_programming","Type_theory",
    "Lambda_calculus","Turing_machine","Halting_problem","Godels_incompleteness_theorems",
    "Set_theory","Category_theory","Model_theory","Proof_theory","Computability_theory",
    "Bayesian_inference","Markov_chain","Random_forest","Support_vector_machine",
    "Gradient_descent","Backpropagation","Transformer_(machine_learning_model)",
    "Attention_mechanism","Recurrent_neural_network","Convolutional_neural_network",
    "Generative_adversarial_network","Variational_autoencoder","Reinforcement_learning",
    "Q-learning","Monte_Carlo_tree_search","AlphaGo","ChatGPT","GPT-4",
    "BERT_(language_model)","Language_model","Tokenization_(lexical_analysis)",
    "Word_embedding","Perplexity","Entropy_(information_theory)","Cross_entropy",
    "Kullback-Leibler_divergence","Softmax_function","Activation_function",
    "Batch_normalization","Dropout_(neural_networks)","Overfitting","Regularization_(mathematics)",
    "Bias-variance_tradeoff","Precision_and_recall","F-score","ROC_curve",
    "Confusion_matrix","K-means_clustering","Principal_component_analysis",
    "Singular_value_decomposition","Eigenvalue","Matrix_(mathematics)",
    "Vector_space","Hilbert_space","Banach_space","Measure_theory","Lebesgue_integration",
    "Real_analysis","Complex_analysis","Functional_analysis","Operator_theory",
    "Stochastic_process","Brownian_motion","Poisson_distribution","Normal_distribution",
    "Central_limit_theorem","Law_of_large_numbers","Hypothesis_testing",
    "Confidence_interval","Regression_analysis","Time_series","Fourier_transform",
    "Laplace_transform","Z-transform","Convolution","Correlation","Autocorrelation",
    "Nyquist-Shannon_sampling_theorem","Data_compression","Huffman_coding",
    "Lempel-Ziv-Welch","Run-length_encoding","Arithmetic_coding",
    "Error_correction_code","Hamming_code","Reed-Solomon_error_correction",
    "Public-key_cryptography","RSA_(cryptosystem)","Elliptic-curve_cryptography",
    "Hash_function","SHA-2","MD5","Blockchain","Bitcoin","Distributed_computing",
    "MapReduce","Apache_Hadoop","Apache_Spark","SQL","NoSQL","B-tree","Hash_table",
    "Linked_list","Binary_tree","Red-black_tree","Heap_(data_structure)",
    "Graph_(discrete_mathematics)","Shortest_path_problem","Maximum_flow_problem",
    "Minimum_spanning_tree","Travelling_salesman_problem","Knapsack_problem",
    "Integer_programming","Linear_programming","Simplex_algorithm","Convex_optimization",
    "Newton's_method","Gradient","Hessian_matrix","Jacobian_matrix_and_determinant",
    "Taylor_series","LHopitals_rule","Riemann_hypothesis","Fermat's_Last_Theorem",
    "P_versus_NP_problem","Four_color_theorem","Poincaré_conjecture",
    "Natural_number","Integer","Rational_number","Real_number","Complex_number",
    "Prime_number","Divisibility","Modular_arithmetic","Chinese_remainder_theorem",
    "Galois_theory","Group_(mathematics)","Ring_(mathematics)","Field_(mathematics)",
    "Polynomial","Permutation","Combination","Pascal's_triangle","Fibonacci_sequence",
    "Golden_ratio","Euler's_number","Pi","Imaginary_number","Quaternion",
    "Tensor","Manifold","Differential_geometry","Riemannian_geometry",
    "Projective_geometry","Euclidean_geometry","Non-Euclidean_geometry",
    "Knot_theory","Algebraic_topology","Homology_(mathematics)","Cohomology",
    "Fiber_bundle","Lie_group","Representation_theory","Harmonic_analysis",
    "Spectral_theory","Distribution_(mathematics)","Sobolev_space","PDE",
    "Fluid_mechanics","Navier-Stokes_equations","Turbulence","Chaos_theory",
    "Dynamical_system","Attractor","Bifurcation_theory","Cellular_automaton",
    "Conway's_Game_of_Life","Boolean_algebra","Logic_gate","Digital_circuit",
    "CPU","GPU","Memory_hierarchy","Cache_(computing)","Virtual_memory",
    "Instruction_set_architecture","RISC-V","x86","ARM_architecture",
    "Parallel_computing","SIMD","OpenMP","MPI","CUDA","GPGPU",
    "Computer_network","TCP/IP","HTTP","DNS","Firewall_(computing)",
    "Cybersecurity","Vulnerability_(computing)","Buffer_overflow","SQL_injection",
    "Malware","Encryption","Digital_signature","Zero-knowledge_proof",
    "Software_engineering","Agile_software_development","DevOps","Version_control",
    "Git","Continuous_integration","Unit_testing","Debugging","Profiling_(computer_programming)",
    "Garbage_collection_(computer_science)","Memory_management","Pointer_(computer_programming)",
    "Stack_(abstract_data_type)","Queue_(abstract_data_type)","Priority_queue",
    "Trie","Bloom_filter","Skip_list","Disjoint-set_data_structure",
    "Segment_tree","Fenwick_tree","Suffix_array","KMP_algorithm",
    "Aho-Corasick_algorithm","Regular_expression","Context-free_grammar","Parsing",
    "Abstract_syntax_tree","Bytecode","Just-in-time_compilation","LLVM",
    "Python_(programming_language)","C_(programming_language)","Rust_(programming_language)",
    "Java_(programming_language)","JavaScript","Haskell_(programming_language)",
    "Prolog","Lisp_(programming_language)","FORTRAN","Assembly_language",
    "Neuron","Synapse","Action_potential","Dopamine","Serotonin",
    "Cortex","Hippocampus","Amygdala","Cerebellum","Prefrontal_cortex",
    "Working_memory","Long-term_memory","Sleep","Consciousness","Perception",
    "Attention","Decision-making","Reward","Motivation","Emotion",
    "Language","Grammar","Syntax","Semantics","Pragmatics","Phonology",
    "DNA","RNA","Protein","Amino_acid","Gene","Chromosome","Cell",
    "Mitochondria","Photosynthesis","Metabolism","Enzyme","Virus","Bacteria",
    "Immune_system","CRISPR","Stem_cell","Cancer","Antibiotic","Vaccine",
    "Atom","Electron","Proton","Neutron","Quark","Boson","Higgs_boson",
    "Standard_Model","Dark_matter","Dark_energy","Big_Bang","Black_hole",
    "Neutron_star","Supernova","Galaxy","Solar_system","Moon","Mars",
    "Climate_change","Greenhouse_effect","Carbon_cycle","Renewable_energy",
    "Solar_panel","Wind_turbine","Battery","Hydrogen","Nuclear_fusion",
    "Internet","World_Wide_Web","Search_engine","Social_media","E-commerce",
    "Smartphone","Transistor","Semiconductor","Moore's_law","Quantum_dot"
)

# Shuffle topics for variety
$Topics = $Topics | Get-Random -Count ([math]::Min($Topics.Count, $ArticleCount))

$total_chars  = 0
$articles_ok  = 0
$articles_fail = 0
$seen_titles  = @{}

Write-Host "=== Wikipedia Corpus Fetcher ==="
Write-Host "Target articles : $ArticleCount"
Write-Host "Output file     : $OutFile"
Write-Host ""

# Open output file (overwrite)
$null = [System.IO.File]::WriteAllText($OutFile, "")

$idx = 0
foreach ($title in $Topics) {
    $idx++
    if ($articles_ok -ge $ArticleCount) { break }

    $url = "https://en.wikipedia.org/api/rest_v1/page/summary/$([uri]::EscapeDataString($title))"
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15 `
                    -Headers @{ "User-Agent" = "pienso-ai-corpus-builder/1.0 (educational)" }
        $json = $resp.Content | ConvertFrom-Json
        $extract = $json.extract

        # Skip disambiguation or missing articles
        if (-not $extract -or $extract.Length -lt 100) {
            $articles_fail++
            continue
        }

        # Skip already-seen canonical titles
        $canon = $json.title
        if ($seen_titles.ContainsKey($canon)) { continue }
        $seen_titles[$canon] = 1

        # Clean: remove parenthetical citations like (word) that are just notes
        $extract = $extract -replace '\([^)]{1,4}\)', ''
        # Collapse multiple spaces
        $extract = $extract -replace '\s+', ' '
        $extract = $extract.Trim()

        # Write: article title as header + paragraph text
        $block = "=== $canon ===`n$extract`n`n"
        [System.IO.File]::AppendAllText($OutFile, $block)

        $total_chars += $extract.Length
        $articles_ok++

        $kb = [math]::Round($total_chars / 1024, 1)
        Write-Host ("  [{0,3}/{1}] {2,-50} {3,6} KB total" -f $articles_ok, $ArticleCount, $canon, $kb)

        Start-Sleep -Milliseconds $DelayMs
    }
    catch {
        $articles_fail++
        # silently skip; network errors are normal
    }
}

Write-Host ""
Write-Host "=== Fetch complete ==="
Write-Host ("Articles fetched : {0}" -f $articles_ok)
Write-Host ("Articles failed  : {0}" -f $articles_fail)
Write-Host ("Total text       : {0:N0} KB" -f ([math]::Round($total_chars / 1024, 1)))
Write-Host ("Output           : $OutFile")

# ── Project Gutenberg plain-text books ────────────────────────────────────
if (-not $SkipGutenberg -and $GutenbergBooks -gt 0) {
    Write-Host ""
    Write-Host "=== Fetching Project Gutenberg books (public domain) ==="

    # Well-known IDs for science/philosophy/math texts (plain UTF-8)
    $GutenbergIds = @(
        84,    # Frankenstein
        1342,  # Pride and Prejudice
        11,    # Alice in Wonderland
        1661,  # Sherlock Holmes
        174,   # Picture of Dorian Gray
        98,    # A Tale of Two Cities
        2701,  # Moby Dick
        1232,  # The Prince (Machiavelli)
        1080,  # A Modest Proposal
        76,    # Adventures of Huckleberry Finn
        5200,  # Metamorphosis (Kafka)
        996,   # Don Quixote (English)
        4300,  # Ulysses
        46,    # A Christmas Carol
        1260,  # Jane Eyre
        244,   # A Study in Scarlet
        43,    # The Strange Case of Dr Jekyll and Mr Hyde
        1400,  # Great Expectations
        2554,  # Crime and Punishment
        600    # Notes from Underground
    ) | Select-Object -First $GutenbergBooks

    foreach ($id in $GutenbergIds) {
        $url = "https://www.gutenberg.org/files/$id/$id-0.txt"
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 30 `
                        -Headers @{ "User-Agent" = "pienso-ai-corpus-builder/1.0 (educational)" }
            $text = $resp.Content

            # Strip Project Gutenberg header/footer boilerplate
            $start = $text.IndexOf("*** START OF")
            $end   = $text.IndexOf("*** END OF")
            if ($start -gt 0) {
                $start = $text.IndexOf("`n", $start) + 1
                if ($end -gt $start) {
                    $text = $text.Substring($start, $end - $start)
                }
            }

            # Take first 50 KB to avoid overwhelming the corpus with one book
            if ($text.Length -gt 51200) { $text = $text.Substring(0, 51200) }
            $text = $text.Trim()

            $block = "=== Gutenberg:$id ===`n$text`n`n"
            [System.IO.File]::AppendAllText($OutFile, $block)

            $kb = [math]::Round($text.Length / 1024, 1)
            Write-Host ("  Gutenberg #{0,-6}  {1,6} KB" -f $id, $kb)
            $total_chars += $text.Length

            Start-Sleep -Milliseconds 500
        }
        catch {
            Write-Host ("  Gutenberg #{0,-6}  FAILED (trying mirror)" -f $id)
            # Try mirror
            try {
                $url2 = "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
                $resp = Invoke-WebRequest -Uri $url2 -UseBasicParsing -TimeoutSec 30 `
                            -Headers @{ "User-Agent" = "pienso-ai-corpus-builder/1.0 (educational)" }
                $text = $resp.Content.Substring(0, [math]::Min($resp.Content.Length, 51200)).Trim()
                $block = "=== Gutenberg:$id ===`n$text`n`n"
                [System.IO.File]::AppendAllText($OutFile, $block)
                Write-Host ("  Gutenberg #{0,-6}  {1,6} KB (mirror)" -f $id, [math]::Round($text.Length/1024,1))
                $total_chars += $text.Length
            }
            catch { }
        }
    }
}

Write-Host ""
Write-Host ("FINAL corpus size: {0:N0} KB  ({1:N2} MB)" -f `
    ([math]::Round($total_chars/1024)), ([math]::Round($total_chars/1048576, 2)))
Write-Host "Done. Next step: run data_prep.exe to deduplicate and filter."
