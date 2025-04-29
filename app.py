from flask import Flask, request, render_template_string, redirect, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import math

app = Flask(__name__)

# Template HTML com campos dinâmicos de restrição e tradução para Português
TEMPLATE = '''
<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8">
    <title>Solucionador de Programação Linear</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px; }
      input, select, button { padding: 5px; margin: 5px 0; }
      .constraint { display: flex; gap: 5px; margin-bottom: 10px; align-items: center; }
      .constraint input, .constraint select { flex: 1; }
      .error { color: red; }
      #add-btn { margin-top: 10px; }
    </style>
    <script>
      function addConstraint() {
        const container = document.getElementById('constraints');
        const div = document.createElement('div');
        div.className = 'constraint';
        div.innerHTML =
          `<input type="number" step="any" name="a" placeholder="a x" required>` +
          `<span>+</span>` +
          `<input type="number" step="any" name="b" placeholder="b y" required>` +
          `<select name="op"><option value="<=">≤</option><option value=">=">≥</option></select>` +
          `<input type="number" step="any" name="c" placeholder="c" required>` +
          `<button type="button" onclick="this.parentNode.remove()">Remover</button>`;
        container.appendChild(div);
      }
      window.onload = function() { addConstraint(); };
    </script>
  </head>
  <body>
    <h1>Solucionador de Programação Linear (2 Variáveis)</h1>
    <form method="post" action="/solve">
      <h2>Função Objetivo</h2>
      Maximizar ou Minimizar?
      <select name="opt">
        <option value="max">Maximizar</option>
        <option value="min">Minimizar</option>
      </select><br/>
      Z = <input type="number" step="any" name="c1" placeholder="Coeficiente de x" required> x +
      <input type="number" step="any" name="c2" placeholder="Coeficiente de y" required> y
      <h2>Restrições</h2>
      <div id="constraints"></div>
      <button type="button" id="add-btn" onclick="addConstraint()">Adicionar Restrição</button>
      <button type="submit">Resolver</button>
    </form>

    {% if error %}
      <p class="error">{{ error }}</p>
    {% endif %}

    {% if result %}
      <h2>Solução</h2>
      <p>Ponto ótimo: ({{ result.x_opt }}, {{ result.y_opt }})</p>
      <p>Valor ótimo Z = {{ result.z_opt }}</p>
      <h2>Método Gráfico</h2>
      <img src="data:image/png;base64,{{ plot_url }}" alt="Região viável">
    {% endif %}
  </body>
</html>
'''


def solve_lp(c, constraints, opt_type='max'):
    points = []
    # interseções com eixos
    for a, b, op, cst in constraints:
        if op == '<=':
            if b != 0:
                points.append((0, cst / b))
            if a != 0:
                points.append((cst / a, 0))
        else:
            if b != 0:
                points.append((0, cst / b))
            if a != 0:
                points.append((cst / a, 0))
    # interseções entre restrições
    n = len(constraints)
    for i in range(n):
        a1, b1, op1, c1 = constraints[i]
        for j in range(i+1, n):
            a2, b2, op2, c2 = constraints[j]
            det = a1*b2 - a2*b1
            if abs(det) > 1e-6:
                x = (c1*b2 - c2*b1) / det
                y = (a1*c2 - a2*c1) / det
                points.append((x, y))
    # filtrar região viável
    feasible = []
    for x, y in points:
        if x < -1e-6 or y < -1e-6:
            continue
        valido = True
        for a, b, op, cst in constraints:
            val = a*x + b*y
            if op == '<=' and val > cst + 1e-6:
                valido = False
                break
            if op == '>=' and val < cst - 1e-6:
                valido = False
                break
        if valido:
            feasible.append((round(x,6), round(y,6)))
    feasible = list(set(feasible))
    if not feasible:
        return None, []
    # avaliar função objetivo
    melhor = None
    for x, y in feasible:
        z = c[0]*x + c[1]*y
        if melhor is None or (opt_type=='max' and z > melhor[2]) or (opt_type=='min' and z < melhor[2]):
            melhor = (x, y, z)
    return melhor, feasible


def plot_region(constraints, feasible, best):
    fig, ax = plt.subplots()
    xmax = max(p[0] for p in feasible)*1.2
    x_vals = np.linspace(0, xmax, 400)
    for a, b, op, cst in constraints:
        if abs(b) > 1e-6:
            y_vals = (cst - a*x_vals) / b
            ax.plot(x_vals, y_vals)
        else:
            x_line = cst / a
            ax.axvline(x_line)
    cx = sum(p[0] for p in feasible)/len(feasible)
    cy = sum(p[1] for p in feasible)/len(feasible)
    sorted_pts = sorted(feasible, key=lambda P: math.atan2(P[1]-cy, P[0]-cx))
    ax.add_patch(plt.Polygon(sorted_pts, alpha=0.3))
    ax.plot(best[0], best[1], 'ro')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max(p[1] for p in feasible)*1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Região Viável e Ponto Ótimo')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')

@app.route('/', methods=['GET'])
def index():
    return render_template_string(TEMPLATE, result=None, error=None)

@app.route('/solve', methods=['GET','POST'])
def solve():
    if request.method == 'GET':
        return redirect(url_for('index'))
    try:
        opt = request.form['opt']
        c = (float(request.form['c1']), float(request.form['c2']))
        constraints = []
        a_vals = request.form.getlist('a')
        b_vals = request.form.getlist('b')
        op_vals = request.form.getlist('op')
        c_vals = request.form.getlist('c')
        for a, b, op, cst in zip(a_vals, b_vals, op_vals, c_vals):
            constraints.append((float(a), float(b), op, float(cst)))
        best, feasible = solve_lp(c, constraints, opt)
        if best is None:
            return render_template_string(TEMPLATE, result=None, error='Nenhuma solução viável.')
        plot_url = plot_region(constraints, feasible, best)
        result = {'x_opt': best[0], 'y_opt': best[1], 'z_opt': round(best[2],4)}
        return render_template_string(TEMPLATE, result=result, plot_url=plot_url, error=None)
    except Exception as e:
        return render_template_string(TEMPLATE, result=None, error=str(e)), 400

if __name__=='__main__':
    app.run(debug=True)
