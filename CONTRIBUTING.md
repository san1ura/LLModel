# Contributing to LLModel

Thank you for your interest in contributing to LLModel! This document provides guidelines for contributing to this project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by the LLModel Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Run the tests to verify your setup:
   ```bash
   pytest
   ```

## Development Workflow

### Branch Strategy
We use a simple branching model:
- `main` - The stable branch with the latest release
- `develop` - The branch for ongoing development
- Feature branches - Branches for specific features or fixes, named as `feature/description` or `fix/description`

### Creating a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-feature
```

## Pull Request Process

1. Ensure your branch is up-to-date with `develop`
   ```bash
   git checkout develop
   git pull origin develop
   git checkout my-feature-branch
   git rebase develop
   ```

2. Ensure all tests pass
   ```bash
   pytest
   ```

3. Update documentation as needed

4. Submit your pull request to the `develop` branch with:
   - A clear title and description
   - References to any related issues
   - Details about any breaking changes

5. Address any feedback from code reviews

## Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings in Google style format
- Keep lines under 100 characters where possible

### Git Commit Messages
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after a blank line

Example:
```
Add tokenizer support for new languages

Resolves #123 by adding support for tokenizing text in
Japanese and Korean languages. This implementation uses
SentencePiece with custom preprocessing for CJK characters.

Fixes #123
```

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected and actual behavior
- Python version and system information
- Any relevant logs or error messages

## First Time Contributors

Look for issues labeled "good first issue" for tasks that are suitable for newcomers to the project. If you have questions, don't hesitate to ask in the issue comments or in our community channels.