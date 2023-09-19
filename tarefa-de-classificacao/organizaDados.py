def organizar_csv(input_filename, output_filename):
    try:
        with open(input_filename, 'r') as input_file:
            lines = input_file.readlines()

        with open(output_filename, 'w') as output_file:
            for i in range(0, len(lines), 1000):
                for j in range(1000):
                    expression_values = ["-1.000000000000000000e+00"] * 5
                    expression_values[i // 1000 % 5] = "1.000000000000000000e+00"
                    output_file.write(','.join(expression_values))
                    if j <= 999:
                        output_file.write('\n')

    except FileNotFoundError:
        print("O arquivo não foi encontrado.")
    except Exception as e:
        print("Ocorreu um erro:", str(e))

organizar_csv('./tarefa-de-classificacao/base-de-dados/EMG.csv', './tarefa-de-classificacao/base-de-dados/labels.csv')
