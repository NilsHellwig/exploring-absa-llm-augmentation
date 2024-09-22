\section{Task Performance: Statistical Tests}\label{appendix:significance-performance}

\begin{table}[H]
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Micro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Micro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline

\multirow{3}{*}{500 real} & \multirow{3}{*}{84.54}  & 500 real + 500 synth & 86.70 & 0.558 & paired t-test (one-sided) & < .01 & < .01 \\
 &  & 500 real + 1,000 synth & 86.60 & 0.952 & paired t-test (one-sided) & < .01 & < .01 \\
 &  & 500 real + 1,500 synth & 85.94 & 0.115 & paired t-test (one-sided) & < .01 & < .01 \\
 \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: ACSA / Metric: F1 micro / LLM: GPT-3.5-turbo / Baseline: 500 real / Scenario: LRS\textsubscript{500}: Augmentation with 500, 1,000 or 1,500 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline
\multirow{3}{*}{500 real} & \multirow{3}{*}{59.52}  & 25 real + 475 synth & 50.86 & 0.200 & paired t-test (one-sided) & 0.988 & 0.988 \\
 &  & 25 real + 975 synth & 62.56 & 0.178 & paired t-test (one-sided) & 0.172 & 0.344 \\
 &  & 25 real + 1,975 synth & 63.67 & 0.401 & paired t-test (one-sided) & 0.035 & 0.106 \\
                        \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: ACSA / Metric: F1 macro / LLM: GPT-3.5-turbo / Baseline: 500 real / Scenario: LRS\textsubscript{25}: Augmentation with 475, 975 or 1,975 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline
\multirow{3}{*}{500 real} & \multirow{3}{*}{59.52}  & 500 real + 500 synth & 70.66 & 0.884 & paired t-test (one-sided) & < .001 & < .001 \\
 &  & 500 real + 1,000 synth & 67.96 & 0.011 & Wilcoxon signed-rank test (one-sided) & 0.031 & 0.031 \\
 &  & 500 real + 1,500 synth & 66.52 & 0.076 & paired t-test (one-sided) & < .01 & < .01 \\
                        \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: ACSA / Metric: F1 macro / LLM: Llama-3-70B / Baseline: 500 real / Scenario: LRS\textsubscript{500}: Augmentation with 500, 1,000 or 1,500 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline
\multirow{3}{*}{500 real} & \multirow{3}{*}{59.52}  & 500 real + 500 synth & 79.00 & 0.734 & paired t-test (one-sided) & < .01 & < .01 \\
 &  & 500 real + 1,000 synth & 78.42 & 0.488 & paired t-test (one-sided) & < .001 & < .001 \\
 &  & 500 real + 1,500 synth & 78.47 & 0.171 & paired t-test (one-sided) & < .001 & < .001 \\
                        \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: ACSA / Metric: F1 macro / LLM: GPT-3.5-turbo / Baseline: 500 real / Scenario: LRS\textsubscript{500}: Augmentation with 500, 1,000 or 1,500 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline
\multirow{3}{*}{1,000 real} & \multirow{3}{*}{74.64}  & 500 real + 500 synth & 79.00 & 0.634 & paired t-test (one-sided) & 0.117 & 0.204 \\
 &  & 500 real + 1,000 synth & 78.42 & 0.686 & paired t-test (one-sided) & 0.102 & 0.204 \\
 &  & 500 real + 1,500 synth & 78.47 & 0.775 & paired t-test (one-sided) & 0.066 & 0.198 \\
                        \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: ACSA / Metric: F1 macro / LLM: GPT-3.5-turbo / Baseline: 1,000 real / Scenario: LRS\textsubscript{500}: Augmentation with 500, 1,000 or 1,500 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline
\multirow{3}{*}{2,000 real} & \multirow{3}{*}{78.86}  & 500 real + 500 synth & 79.00 & 0.870 & paired t-test (one-sided) & 0.480 & 1.000 \\
 &  & 500 real + 1,000 synth & 78.42 & 0.620 & paired t-test (one-sided) & 0.584 & 1.000 \\
 &  & 500 real + 1,500 synth & 78.47 & 0.669 & paired t-test (one-sided) & 0.592 & 1.000 \\
                        \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: ACSA / Metric: F1 macro / LLM: GPT-3.5-turbo / Baseline: 2,000 real / Scenario: LRS\textsubscript{500}: Augmentation with 500, 1,000 or 1,500 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\begin{subtable}{\linewidth}
\centering
\scriptsize
\resizebox{1.0\columnwidth}{!}{%
\begin{tabular}{llllllll}
\hline
\multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Baseline\end{tabular}}}                                                                                                                     & \multicolumn{2}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Group Augmentation\end{tabular}}}                                                                                                                 & \multicolumn{1}{c}{\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}p Normality Test \\ (Shapiro-Wilk)\end{tabular}}}} & \multicolumn{3}{c}{\textbf{Test for Significance}}                                                     \\ \cline{1-4} \cline{6-8} 
\multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Training \\ Examples\end{tabular}}} & \multicolumn{1}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Mean F1 Macro\\ Across All Six Iterations\end{tabular}}} & \multicolumn{1}{c}{}                                                                                                     & \multicolumn{1}{l}{\textbf{Name}} & \multicolumn{1}{c}{\textbf{p}} & \multicolumn{1}{c}{\textbf{p\textsubscript{adj}}} \\ \hline
\multirow{3}{*}{500 real} & \multirow{3}{*}{70.07}  & 500 real + 500 synth & 72.27 & 0.479 & paired t-test (one-sided) & 0.020 & 0.059 \\
 &  & 500 real + 1,000 synth & 72.55 & 0.438 & paired t-test (one-sided) & 0.175 & 0.350 \\
 &  & 500 real + 1,500 synth & 71.67 & 0.253 & paired t-test (one-sided) & 0.261 & 0.350 \\
                        \hline
\end{tabular}
}
\vspace{0.1cm}
\caption{Task: E2E-ABSA / Metric: F1 macro / LLM: GPT-3.5-turbo / Baseline: 500 real / Scenario: LRS\textsubscript{500}: Augmentation with 500, 1,000 or 1,500 synthetic examples}
\vspace{0.3cm}
\end{subtable}
\caption{Results of the statistical tests to check for significant improvements by adding synthetic examples to the training}
\label{fig:significance-performance}
\end{table}
