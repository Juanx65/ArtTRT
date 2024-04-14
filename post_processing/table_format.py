def generate_latex_table(data):
    latex_header = r"""
\begin{table}[h]
    \centering
    \footnotesize
    \begin{tabular}{@{}ccccccccccc@{}}
    \toprule
    & & & \multicolumn{2}{c|}{Bs 1} & \multicolumn{1}{c|}{Bs 256} & & & & & \\ \midrule
    & & & \textbf{La ms} & \multicolumn{1}{c|}{\textbf{Lwc ms}} & \multicolumn{1}{c|}{\textbf{Ta inf/s}} & \textbf{Tm Mb} & \textbf{C} & \textbf{P} & \textbf{Top 1 \%} & \textbf{Top 5 \%} \\ \midrule
    """
    latex_footer = r"""
    \end{tabular}
\end{table}
    """

    latex_content = ""
    first_loop = True
    for model_name, variants in data.items():
        if variants:
            if first_loop:
                model_section = f" \\multirow{{24}}{{*}}{{\\rotatebox[origin=c]{{90}}{{Desktop}}}} "
                first_loop = False
            else:
                model_section = f" "
            for i, variant in enumerate(variants):
                variant_name = variant['variant'].replace('_', ' ')  # Replace underscores with spaces
                if i == 0:
                    model_section += f"& \\multirow{{4}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{model_name}}}}}  "
                    model_section += f"& \\textit{{{variant_name}}} & {variant['lat_mean']} & \\multicolumn{{1}}{{c|}}{{{variant['lat_worst']}}} & \\multicolumn{{1}}{{c|}}{{{variant['inf_s']}}} & {variant['size_mb']} & {variant['layers']} & {variant['params']} & {variant['prec_1']} & {variant['prec_5']} \\\\"
                else:
                    model_section += f"    & & \\textit{{{variant_name}}} & {variant['lat_mean']} & \\multicolumn{{1}}{{c|}}{{{variant['lat_worst']}}} & \\multicolumn{{1}}{{c|}}{{{variant['inf_s']}}} & {variant['size_mb']} & {variant['layers']} & {variant['params']} & {variant['prec_1']} & {variant['prec_5']} \\\\"
                
                if i == len(variants) - 1:
                    model_section += " \\cmidrule(l){2-11} \n"
                else:
                    model_section += "\n"
            latex_content += model_section

    full_latex = latex_header + latex_content + latex_footer
    return full_latex



def parse_markdown_data(file_content):
    models_data = file_content.split('# ')[1:]  # Divide por modelo y batch size, ignorando el primer split que estaría vacío
    data_summary = {}
    inf_s_data = {}

    # Procesar y almacenar inf/s para batch size 256 por separado
    for model_section in models_data:
        lines = model_section.split('\n')
        model_name_batch_size = lines[0].strip()
        model_name, batch_size_info = model_name_batch_size.split(' bs ')
        batch_size = int(batch_size_info.strip())

        if batch_size == 256:
            for line in lines:
                if '|' in line and 'Model' not in line and not all(char == '-' for char in line.replace('|', '').strip()):
                    cols = [col.strip() for col in line.split('|')]
                    if len(cols) > 1:
                        variant = cols[1]
                        inf_s = cols[2].split(' ')[0]
                        if model_name not in inf_s_data:
                            inf_s_data[model_name] = {}
                        inf_s_data[model_name][variant] = inf_s

    # Procesar datos principales para batch size 1 y añadir inf/s de batch size 256 cuando corresponda
    for model_section in models_data:
        lines = model_section.split('\n')
        model_name_batch_size = lines[0].strip()
        model_name, batch_size_info = model_name_batch_size.split(' bs ')
        batch_size = int(batch_size_info.strip())

        if batch_size == 1:
            table_data = [line for line in lines if '|' in line and 'Model' not in line]

            data_summary[model_name] = []

            for data in table_data:
                if not all(char == '-' for char in data.replace('|', '').strip()):
                    cols = [col.strip() for col in data.split('|')]
                    if len(cols) > 1:
                        model_variant = cols[1]
                        inf_s = inf_s_data.get(model_name, {}).get(model_variant, 'N/A')  # Get inf/s from bs 256 data if available
                        lat_mean = cols[3].split(' / ')[0]
                        lat_worst = (cols[3].split(' / ')[1]).split(' ')[0]
                        lat_mean = lat_mean.strip()
                        lat_worst = lat_worst.strip()
                        size_mb = cols[4]
                        prec_1 = cols[5]
                        prec_5 = cols[6]
                        layers = cols[7]
                        params = cols[8]

                        model_data = {
                            'variant': model_variant,
                            'inf_s': inf_s,  # Added from bs 256 data
                            'lat_mean': lat_mean,
                            'lat_worst': lat_worst,
                            'size_mb': size_mb.strip(),
                            'prec_1': prec_1.strip(),
                            'prec_5': prec_5.strip(),
                            'layers': layers.strip(),
                            'params': params.strip()
                        }
                        data_summary[model_name].append(model_data)

    return data_summary


file_path = '../outputs/table_outputs/rtx3060/rtx3060_dynamic_batch.md'

# Abre y lee el archivo
with open(file_path, 'r', encoding='utf-8') as file:
    md_content = file.read()
# Parse the markdown content
parsed_data = parse_markdown_data(md_content)

# Generate LaTeX table
latex_table = generate_latex_table(parsed_data)
print(latex_table)
