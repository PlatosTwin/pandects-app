# Pandects App

**Pandects** is an open-source M\&A research platform built with a Vite frontend and Flask backend.

For more on the project, including data sources and pipelines, see our [About page](https://pandects-app.fly.dev/about) on the frontend.

We welcome contributions to the frontend via pull requests. To contribute to data pipelines and analytics, please send a note to Nikita Bogdanov at nmbogdan [at] alumni [dot] stanford [dot] edu.

## Getting Started

### Prerequisites

* Python 3.10+
* Node.js 18+

### Installation

```bash
# Clone the repo
git clone https://github.com/PlatosTwin/pandects-app.git
cd pandects-app
```

#### Backend (Flask)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
flask run
```

#### Frontend (Vite + React/TypeScript)

```bash
cd frontend
npm install
npm run dev
```

Navbar easter egg (purely cosmetic): type `panda` anywhere (when not focused in a text input) to toggle a little gravity-driven ball.

```bash
export VITE_DISABLE_PANDA_EASTER_EGG=1
```

Optional: switch the end effect back to a fade-out:

```bash
export VITE_PANDA_END_STYLE=fade
```

## Contributing

### Reporting Bugs & Requesting Features

1. **Search existing issues** to see if your bug or feature request already exists.
2. **Open a new issue** and include:

   * A clear, descriptive title.
   * A concise description of the problem or desired enhancement.
   * Steps to reproduce (for bugs) or example usage (for features).
   * Any relevant logs or screenshots.

### Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:

   ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/pandects-app.git
    cd pandects-app
    ```
3. **Create a branch** for your work:  
    ```bash
    git checkout -b feature/new-thing
    ```

### Submitting a Pull Request

1. **Push** your branch to your fork:

   ```bash
   git push origin feature/new-thing
   ```

2. **Open a Pull Request** against `main` in the upstream repository.
3. In your PR description, link to any related issues and describe:
   - What you’ve changed
   - Why it’s needed
   - How to test it

## License

This project is licensed under the GNU GPLv3 license. See [LICENSE](LICENSE) for details.
