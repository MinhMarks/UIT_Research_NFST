import re
import os

# Use path relative to this script's location so it works on any machine/server
_script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(_script_dir, 'main.tex')

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Rename Section 4 to Experimental Evaluation
text = text.replace(r'\section{Experiment Set Up}', r'\section{Experimental Evaluation}')

# 2. Add \subsection{Experimental Setup} before \subsubsection{Datasets}
text = text.replace(r'\subsection{Datasets}', r'\subsection{Experimental Setup}' + '\n' + r'\subsubsection{Datasets}')

# 3. Change subsections to subsubsections under Experimental Setup
text = text.replace(r'\subsection{Experiment Training and Testing Strategy}', r'\subsubsection{Experiment Training and Testing Strategy}')
text = text.replace(r'\subsection{Baselines and Competitors}', r'\subsubsection{Baselines and Competitors}')
text = text.replace(r'\subsection{Evaluation Metrics}', r'\subsubsection{Evaluation Metrics}')
text = text.replace(r'\subsection{ Software and hardware environment configuration}', r'\subsubsection{Software and hardware environment configuration}')

# 4. Remove the redundant \section{Experiment results and competitor} and replace with \subsection{Results and Analysis} 
# and absorb the text of \subsection{Experiment Results}
text = re.sub(
    r'\\section\{Experiment results and competitor\}(.*?)\\subsection\{Experiment Results\}(.*?)\\label\{subsec:exp_rs\}',
    r'\\subsection{Results and Analysis}\1\n\\label{subsec:exp_rs}',
    text,
    flags=re.DOTALL
)

# 5. Rename specific subsubsections for clarity
text = text.replace(r'\subsubsection{Anomaly Detection Performance on 4 Datasets IoT (RQ1)}', r'\subsubsection{Detection Performance on IoT Benchmarks (RQ1)}')
text = text.replace(r'\subsubsection{Detection Performance on Different Anomaly Types and Noise Levels}', r'\subsubsection{Robustness Against Contamination and Anomaly Types (RQ2)}')

# 6. Change "Runtime Efficiency and Scalability (RQ3)" -> "Computational Efficiency and Memory Trade-offs (RQ3)"
text = text.replace(r'\subsubsection{Runtime Efficiency and Scalability (RQ3)}', r'\subsubsection{Computational Efficiency and Memory Trade-offs (RQ3)}')

# 7. Inject the memory plots content right after the radar chart in the RQ3 section
radar_chart_code = r'''\begin{figure}[!ht]
\centering  
\includegraphics[width=\columnwidth]{pictures/radar_chart.png} 
\caption{Comparison of standardized training and testing ranks, and train/test times between XXX and baseline methods. The blue line represents XXX, while the orange shaded area indicates the median performance of baselines.}
\label{fig:radar_chart}
\end{figure}'''

memory_plots_code = r'''

Furthermore, we explicitly evaluate the memory and computational trade-offs of the XXX algorithm. Figure~\ref{fig:memory_footprint} visualizes the peak RAM usage during training for both XXX and the baseline methods.

\begin{figure}[!ht]
\centering
\includegraphics[width=\columnwidth]{notebooks/analysis/memory_comparison/plots/Bar_Memory_Footprint.png} 
\caption{Comparison of Peak RAM usage (MB) during training across various models, highlighting the memory efficiency of XXX.}
\label{fig:memory_footprint}
\end{figure}

To further illustrate the balance between detection performance and memory overhead, Figure~\ref{fig:efficiency_tradeoff} presents an efficiency trade-off scatter plot. The plot contrasts the AUC-PR scores against the peak RAM consumption, demonstrating that XXX achieves state-of-the-art detection accuracy while maintaining a constrained memory footprint, making it highly suitable for IoT edge deployments.

\begin{figure}[!ht]
\centering
\includegraphics[width=\columnwidth]{notebooks/analysis/memory_comparison/plots/Scatter_Efficiency_Tradeoff.png} 
\caption{Efficiency vs. Performance Trade-off: AUC-PR vs. Peak RAM usage.}
\label{fig:efficiency_tradeoff}
\end{figure}

Finally, Figure~\ref{fig:time_complexity} compares the computational time complexity (Train and Test Time) on a logarithmic scale, reassuring that XXX remains competitive in runtime speed against standard lightweight baselines while significantly outperforming deep learning architectures.

\begin{figure}[!ht]
\centering
\includegraphics[width=\columnwidth]{notebooks/analysis/memory_comparison/plots/Bar_Time_Complexity.png} 
\caption{Train and Test Time complexity comparison on a logarithmic scale.}
\label{fig:time_complexity}
\end{figure}
'''

if radar_chart_code in text:
    text = text.replace(radar_chart_code, radar_chart_code + memory_plots_code)
else:
    print("Warning: Could not find radar chart to inject memory plots after.")

# 8. Rename Competior subsection -> Comparison with State-of-the-Art
text = text.replace(r'\subsection{Competior}', r'\subsubsection{Comparison with State-of-the-Art}')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Successfully applied restructuring to {filepath}")
