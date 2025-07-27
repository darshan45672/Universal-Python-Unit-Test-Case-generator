#!/usr/bin/env python3
"""
Universal Python Unit Test Generator AI Agent
A comprehensive AI-powered system for generating precise unit tests for Python codebases.
"""

import ast
import os
import sys
import json
import time
import asyncio
import argparse
import logging
import hashlib
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum
import importlib.util
import re
from concurrent.futures import ThreadPoolExecutor
import threading

# Third-party imports (with fallback handling)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class LogLevel(Enum):
    """Logging levels for the agent."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"


class ProjectType(Enum):
    """Types of Python projects."""
    SIMPLE_SCRIPT = "simple_script"
    DATA_HANDLER = "data_handler"
    CRUD_APP = "crud_app"
    FRAMEWORK_HEAVY = "framework_heavy"
    LIBRARY = "library"


@dataclass
class TestGenerationConfig:
    """Configuration for test generation."""
    target_repo: Path
    test_framework: TestFramework = TestFramework.PYTEST
    llm_provider: LLMProvider = LLMProvider.OPENAI
    api_key: Optional[str] = None
    dry_run: bool = False
    verbose: bool = False
    use_cache: bool = True
    max_retries: int = 3
    timeout: int = 30
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", ".pytest_cache", "venv", "env", ".env",
        "node_modules", "dist", "build", "*.egg-info", "tests", "test_*"
    ])


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    path: Path
    name: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    complexity_score: int
    project_type: ProjectType
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of test generation and validation."""
    module_name: str
    test_file_path: Path
    tests_generated: int
    tests_passed: int
    tests_failed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Logger:
    """Enhanced logging system."""
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.lock = threading.Lock()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, level.value),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('testgen.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def debug(self, message: str):
        with self.lock:
            self.logger.debug(message)
    
    def info(self, message: str):
        with self.lock:
            self.logger.info(message)
    
    def warning(self, message: str):
        with self.lock:
            self.logger.warning(message)
    
    def error(self, message: str):
        with self.lock:
            self.logger.error(message)


class CacheManager:
    """Caching system for LLM responses."""
    
    def __init__(self, cache_dir: Path = Path(".testgen_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
    
    def _get_cache_key(self, prompt: str, provider: str) -> str:
        """Generate cache key from prompt and provider."""
        content = f"{provider}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, provider: str) -> Optional[str]:
        """Get cached response."""
        with self.lock:
            cache_key = self._get_cache_key(prompt, provider)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        # Check if cache is still valid (24 hours)
                        if time.time() - data['timestamp'] < 86400:
                            return data['response']
                except Exception:
                    pass
        return None
    
    def set(self, prompt: str, provider: str, response: str):
        """Cache response."""
        with self.lock:
            cache_key = self._get_cache_key(prompt, provider)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'timestamp': time.time(),
                        'response': response
                    }, f)
            except Exception as e:
                pass  # Fail silently for cache errors


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    @abstractmethod
    async def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if HAS_OPENAI and api_key:
            openai.api_key = api_key
    
    def is_available(self) -> bool:
        return HAS_OPENAI and self.api_key is not None
    
    async def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        if not self.is_available():
            raise Exception("OpenAI backend not available")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class GeminiBackend(LLMBackend):
    """Google Gemini backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if HAS_GEMINI and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
    
    def is_available(self) -> bool:
        return HAS_GEMINI and self.api_key is not None
    
    async def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        if not self.is_available():
            raise Exception("Gemini backend not available")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


class ClaudeBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if HAS_ANTHROPIC and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def is_available(self) -> bool:
        return HAS_ANTHROPIC and self.api_key is not None
    
    async def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        if not self.is_available():
            raise Exception("Claude backend not available")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "codellama"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def is_available(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        if not self.is_available():
            raise Exception("Ollama backend not available")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60
                )
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")


class LLMFactory:
    """Factory for creating LLM backends."""
    
    @staticmethod
    def create_backend(provider: LLMProvider, api_key: Optional[str] = None) -> LLMBackend:
        """Create LLM backend based on provider."""
        backends = {
            LLMProvider.OPENAI: OpenAIBackend,
            LLMProvider.GEMINI: GeminiBackend,
            LLMProvider.CLAUDE: ClaudeBackend,
            LLMProvider.OLLAMA: OllamaBackend,
        }
        
        backend_class = backends.get(provider)
        if not backend_class:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return backend_class(api_key)


class PythonAnalyzer:
    """Analyzes Python code structure and complexity."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def analyze_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a Python file and extract relevant information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract information
            imports = self._extract_imports(tree)
            classes = self._extract_classes(tree)
            functions = self._extract_functions(tree)
            complexity = self._calculate_complexity(tree)
            project_type = self._determine_project_type(content, imports)
            
            return ModuleInfo(
                path=file_path,
                name=file_path.stem,
                imports=imports,
                classes=classes,
                functions=functions,
                complexity_score=complexity,
                project_type=project_type
            )
        
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            return None
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _extract_classes(self, tree: ast.AST) -> List[str]:
        """Extract class definitions."""
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """Extract function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods and test functions
                if not node.name.startswith('_') and not node.name.startswith('test_'):
                    functions.append(node.name)
        return functions
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate code complexity score."""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += len(node.args.args)
        return complexity
    
    def _determine_project_type(self, content: str, imports: List[str]) -> ProjectType:
        """Determine the type of Python project."""
        # Framework detection
        web_frameworks = ['flask', 'django', 'fastapi', 'tornado', 'bottle']
        data_frameworks = ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn']
        
        if any(fw in imports for fw in web_frameworks):
            return ProjectType.CRUD_APP
        elif any(df in imports for df in data_frameworks):
            return ProjectType.DATA_HANDLER
        elif len(imports) > 10 or 'class' in content.lower():
            return ProjectType.FRAMEWORK_HEAVY
        elif any(keyword in content.lower() for keyword in ['def ', 'class ']):
            return ProjectType.LIBRARY
        else:
            return ProjectType.SIMPLE_SCRIPT


class TestGenerationStrategy(ABC):
    """Abstract strategy for test generation."""
    
    @abstractmethod
    async def generate_tests(self, module_info: ModuleInfo, llm_backend: LLMBackend) -> str:
        """Generate test code for the module."""
        pass


class SimpleScriptStrategy(TestGenerationStrategy):
    """Strategy for simple scripts."""
    
    async def generate_tests(self, module_info: ModuleInfo, llm_backend: LLMBackend) -> str:
        prompt = f"""
Generate comprehensive unit tests for a simple Python script with the following information:

Module: {module_info.name}
Functions: {', '.join(module_info.functions)}
Imports: {', '.join(module_info.imports)}

Requirements:
1. Use pytest framework
2. Test all functions with valid inputs, edge cases, and error cases
3. Include proper mocking for external dependencies
4. Add docstrings to test functions
5. Handle potential exceptions gracefully

Generate only the test code, no explanations.
"""
        return await llm_backend.generate_response(prompt)


class CRUDAppStrategy(TestGenerationStrategy):
    """Strategy for CRUD applications."""
    
    async def generate_tests(self, module_info: ModuleInfo, llm_backend: LLMBackend) -> str:
        prompt = f"""
Generate comprehensive unit tests for a CRUD application module with the following information:

Module: {module_info.name}
Classes: {', '.join(module_info.classes)}
Functions: {', '.join(module_info.functions)}
Imports: {', '.join(module_info.imports)}

Requirements:
1. Use pytest framework with fixtures
2. Mock database connections and external APIs
3. Test CRUD operations (Create, Read, Update, Delete)
4. Test authentication and authorization if present
5. Test input validation and error handling
6. Include integration-style tests for endpoints
7. Use proper test database setup and teardown

Generate only the test code, no explanations.
"""
        return await llm_backend.generate_response(prompt)


class FrameworkHeavyStrategy(TestGenerationStrategy):
    """Strategy for framework-heavy applications."""
    
    async def generate_tests(self, module_info: ModuleInfo, llm_backend: LLMBackend) -> str:
        prompt = f"""
Generate comprehensive unit tests for a complex framework-heavy module with the following information:

Module: {module_info.name}
Classes: {', '.join(module_info.classes)}
Functions: {', '.join(module_info.functions)}
Imports: {', '.join(module_info.imports)}
Complexity Score: {module_info.complexity_score}

Requirements:
1. Use pytest framework with advanced fixtures
2. Mock complex dependencies and external services
3. Test class hierarchies and inheritance
4. Test async/await patterns if present
5. Include parametrized tests for multiple scenarios
6. Test error handling and edge cases
7. Use dependency injection mocking
8. Include performance considerations

Generate only the test code, no explanations.
"""
        return await llm_backend.generate_response(prompt)


class TestGenerationStrategyFactory:
    """Factory for creating test generation strategies."""
    
    @staticmethod
    def create_strategy(project_type: ProjectType) -> TestGenerationStrategy:
        """Create strategy based on project type."""
        strategies = {
            ProjectType.SIMPLE_SCRIPT: SimpleScriptStrategy,
            ProjectType.DATA_HANDLER: SimpleScriptStrategy,  # Reuse simple strategy
            ProjectType.CRUD_APP: CRUDAppStrategy,
            ProjectType.FRAMEWORK_HEAVY: FrameworkHeavyStrategy,
            ProjectType.LIBRARY: FrameworkHeavyStrategy,  # Reuse framework strategy
        }
        
        strategy_class = strategies.get(project_type, SimpleScriptStrategy)
        return strategy_class()


class TestValidator:
    """Validates and fixes generated tests."""
    
    def __init__(self, logger: Logger, config: TestGenerationConfig):
        self.logger = logger
        self.config = config
    
    async def validate_and_fix_tests(self, test_code: str, module_info: ModuleInfo, 
                                   llm_backend: LLMBackend) -> Tuple[str, List[str]]:
        """Validate tests and attempt to fix issues."""
        errors = []
        
        # Create temporary test file
        temp_test_file = Path(f"temp_test_{module_info.name}.py")
        
        try:
            # Write test code to temporary file
            with open(temp_test_file, 'w') as f:
                f.write(test_code)
            
            # Try to import and run basic syntax check
            spec = importlib.util.spec_from_file_location("temp_test", temp_test_file)
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    errors.append(f"Import error: {str(e)}")
            
            # If dry run is disabled, try running the tests
            if not self.config.dry_run:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", str(temp_test_file), "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    errors.append(f"Test execution failed: {result.stderr}")
            
            # If there are errors, try to fix them
            if errors and len(errors) <= 3:  # Only try to fix if not too many errors
                fixed_code = await self._fix_test_code(test_code, errors, llm_backend)
                if fixed_code and fixed_code != test_code:
                    # Validate the fixed code
                    with open(temp_test_file, 'w') as f:
                        f.write(fixed_code)
                    
                    try:
                        spec = importlib.util.spec_from_file_location("temp_test", temp_test_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                        return fixed_code, []  # Fixed successfully
                    except Exception as e:
                        errors.append(f"Fix attempt failed: {str(e)}")
            
            return test_code, errors
        
        except Exception as e:
            self.logger.error(f"Test validation error: {str(e)}")
            return test_code, [str(e)]
        
        finally:
            # Clean up temporary file
            if temp_test_file.exists():
                temp_test_file.unlink()
    
    async def _fix_test_code(self, test_code: str, errors: List[str], 
                           llm_backend: LLMBackend) -> Optional[str]:
        """Attempt to fix test code based on errors."""
        prompt = f"""
Fix the following Python test code that has these errors:

Errors:
{chr(10).join(errors)}

Test Code:
{test_code}

Please fix the issues and return only the corrected test code, no explanations.
"""
        
        try:
            return await llm_backend.generate_response(prompt, max_tokens=3000)
        except Exception as e:
            self.logger.error(f"Failed to fix test code: {str(e)}")
            return None


class UniversalTestGenerator:
    """Main test generator class implementing the Observer pattern."""
    
    def __init__(self, config: TestGenerationConfig):
        self.config = config
        self.logger = Logger(LogLevel.DEBUG if config.verbose else LogLevel.INFO)
        self.cache_manager = CacheManager() if config.use_cache else None
        self.analyzer = PythonAnalyzer(self.logger)
        self.validator = TestValidator(self.logger, config)
        
        # Initialize LLM backend
        self.llm_backend = LLMFactory.create_backend(config.llm_provider, config.api_key)
        
        # Results tracking
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
    
    async def generate_tests_for_repository(self) -> List[TestResult]:
        """Generate tests for the entire repository."""
        self.logger.info(f"Starting test generation for repository: {self.config.target_repo}")
        
        # Discover Python files
        python_files = self._discover_python_files()
        self.logger.info(f"Found {len(python_files)} Python files to process")
        
        # Create tests directory
        tests_dir = self.config.target_repo / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Create __init__.py in tests directory
        (tests_dir / "__init__.py").touch()
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(3)  # Limit concurrent LLM calls
        tasks = [self._process_file_with_semaphore(semaphore, file_path, tests_dir) 
                for file_path in python_files]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Generate pytest configuration
        self._generate_pytest_config(tests_dir)
        
        self.logger.info(f"Test generation completed. Generated {len(self.results)} test files.")
        return self.results
    
    async def _process_file_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                         file_path: Path, tests_dir: Path):
        """Process a file with concurrency control."""
        async with semaphore:
            await self._process_file(file_path, tests_dir)
    
    def _discover_python_files(self) -> List[Path]:
        """Discover Python files in the repository."""
        python_files = []
        
        for root, dirs, files in os.walk(self.config.target_repo):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(
                pattern in d for pattern in self.config.exclude_patterns
            )]
            
            for file in files:
                if file.endswith('.py') and not any(
                    pattern in file for pattern in self.config.exclude_patterns
                ):
                    file_path = Path(root) / file
                    # Skip if it's this generator script
                    if file_path.name != Path(__file__).name:
                        python_files.append(file_path)
        
        return python_files
    
    async def _process_file(self, file_path: Path, tests_dir: Path):
        """Process a single Python file."""
        try:
            self.logger.info(f"Processing: {file_path}")
            
            # Analyze the file
            module_info = self.analyzer.analyze_file(file_path)
            if not module_info:
                self.logger.warning(f"Skipping {file_path}: Analysis failed")
                return
            
            # Skip if no testable content
            if not module_info.functions and not module_info.classes:
                self.logger.info(f"Skipping {file_path}: No testable functions or classes")
                return
            
            # Generate test code
            test_code = await self._generate_test_code(module_info)
            if not test_code:
                self.logger.warning(f"Failed to generate tests for {file_path}")
                return
            
            # Validate and fix tests
            validated_code, errors = await self.validator.validate_and_fix_tests(
                test_code, module_info, self.llm_backend
            )
            
            # Write test file
            test_file_path = tests_dir / f"test_{module_info.name}.py"
            with open(test_file_path, 'w') as f:
                f.write(validated_code)
            
            # Calculate test statistics
            test_count = validated_code.count('def test_')
            passed_count = test_count - len(errors)
            
            # Record result
            result = TestResult(
                module_name=module_info.name,
                test_file_path=test_file_path,
                tests_generated=test_count,
                tests_passed=passed_count,
                tests_failed=len(errors),
                errors=errors
            )
            
            with self.lock:
                self.results.append(result)
            
            self.logger.info(f"Generated {test_count} tests for {module_info.name}")
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
    
    async def _generate_test_code(self, module_info: ModuleInfo) -> Optional[str]:
        """Generate test code for a module."""
        # Check cache first
        cache_key = f"{module_info.name}_{module_info.project_type.value}"
        if self.cache_manager:
            cached_response = self.cache_manager.get(cache_key, self.config.llm_provider.value)
            if cached_response:
                self.logger.debug(f"Using cached response for {module_info.name}")
                return cached_response
        
        # Generate using strategy pattern
        strategy = TestGenerationStrategyFactory.create_strategy(module_info.project_type)
        
        # Try with retries
        for attempt in range(self.config.max_retries):
            try:
                test_code = await asyncio.wait_for(
                    strategy.generate_tests(module_info, self.llm_backend),
                    timeout=self.config.timeout
                )
                
                # Cache the response
                if self.cache_manager and test_code:
                    self.cache_manager.set(cache_key, self.config.llm_provider.value, test_code)
                
                return test_code
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {module_info.name}: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _generate_pytest_config(self, tests_dir: Path):
        """Generate pytest configuration file."""
        config_content = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
"""
        
        config_file = self.config.target_repo / "pytest.ini"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        self.logger.info("Generated pytest.ini configuration")
    
    def print_summary(self):
        """Print generation summary."""
        total_tests = sum(r.tests_generated for r in self.results)
        total_passed = sum(r.tests_passed for r in self.results)
        total_failed = sum(r.tests_failed for r in self.results)
        
        print("\n" + "="*60)
        print("TEST GENERATION SUMMARY")
        print("="*60)
        print(f"Files processed: {len(self.results)}")
        print(f"Total tests generated: {total_tests}")
        print(f"Tests passed: {total_passed}")
        print(f"Tests failed: {total_failed}")
        print(f"Success rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print("="*60)
        
        if self.config.verbose:
            print("\nDETAILED RESULTS:")
            print("-" * 40)
            for result in self.results:
                print(f"Module: {result.module_name}")
                print(f"  Tests: {result.tests_generated} | Passed: {result.tests_passed} | Failed: {result.tests_failed}")
                if result.errors:
                    print(f"  Errors: {'; '.join(result.errors[:2])}")
                print()


class InteractiveCLI:
    """Interactive command-line interface."""
    
    def __init__(self):
        self.config = TestGenerationConfig(target_repo=Path.cwd())
    
    def run_interactive_mode(self):
        """Run interactive configuration mode."""
        print("=" * 60)
        print("Universal Python Unit Test Generator")
        print("=" * 60)
        
        # Get target repository
        repo_path = input(f"Enter target repository path [{self.config.target_repo}]: ").strip()
        if repo_path:
            self.config.target_repo = Path(repo_path)
        
        if not self.config.target_repo.exists():
            print(f"Error: Repository path does not exist: {self.config.target_repo}")
            return False
        
        # Select LLM provider
        print("\nAvailable LLM providers:")
        providers = list(LLMProvider)
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider.value}")
        
        while True:
            try:
                choice = int(input(f"Select LLM provider [1-{len(providers)}]: "))
                if 1 <= choice <= len(providers):
                    self.config.llm_provider = providers[choice - 1]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get API key if needed
        if self.config.llm_provider != LLMProvider.OLLAMA:
            api_key = input(f"Enter API key for {self.config.llm_provider.value}: ").strip()
            if not api_key:
                print("Warning: No API key provided. This may cause authentication errors.")
            self.config.api_key = api_key or None
        
        # Select test framework
        print("\nAvailable test frameworks:")
        frameworks = list(TestFramework)
        for i, framework in enumerate(frameworks, 1):
            print(f"{i}. {framework.value}")
        
        while True:
            try:
                choice = int(input(f"Select test framework [1-{len(frameworks)}]: "))
                if 1 <= choice <= len(frameworks):
                    self.config.test_framework = frameworks[choice - 1]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Additional options
        self.config.verbose = input("Enable verbose output? [y/N]: ").lower().startswith('y')
        self.config.dry_run = input("Run in dry-run mode (no test execution)? [y/N]: ").lower().startswith('y')
        self.config.use_cache = not input("Disable caching? [y/N]: ").lower().startswith('y')
        
        return True
    
    def display_config(self):
        """Display current configuration."""
        print("\nConfiguration:")
        print("-" * 30)
        print(f"Repository: {self.config.target_repo}")
        print(f"LLM Provider: {self.config.llm_provider.value}")
        print(f"Test Framework: {self.config.test_framework.value}")
        print(f"Verbose: {self.config.verbose}")
        print(f"Dry Run: {self.config.dry_run}")
        print(f"Use Cache: {self.config.use_cache}")
        print("-" * 30)


class CLIArgumentParser:
    """Command-line argument parser."""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Universal Python Unit Test Generator AI Agent",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python testgen.py --interactive
  python testgen.py --model openai --api-key YOUR_KEY --repo ./my_project
  python testgen.py --model ollama --repo ./my_project --framework pytest --verbose
  python testgen.py --model gemini --api-key YOUR_KEY --dry-run --no-cache
            """
        )
        self._setup_arguments()
    
    def _setup_arguments(self):
        """Setup command-line arguments."""
        # Mode selection
        mode_group = self.parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Run in interactive mode'
        )
        
        # LLM configuration
        self.parser.add_argument(
            '--model', '-m',
            type=str,
            choices=[p.value for p in LLMProvider],
            default=LLMProvider.OPENAI.value,
            help='LLM provider to use (default: openai)'
        )
        
        self.parser.add_argument(
            '--api-key', '-k',
            type=str,
            help='API key for the selected LLM provider'
        )
        
        # Repository configuration
        self.parser.add_argument(
            '--repo', '-r',
            type=Path,
            default=Path.cwd(),
            help='Target repository path (default: current directory)'
        )
        
        # Test framework
        self.parser.add_argument(
            '--framework', '-f',
            type=str,
            choices=[f.value for f in TestFramework],
            default=TestFramework.PYTEST.value,
            help='Test framework to use (default: pytest)'
        )
        
        # Execution options
        self.parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Generate tests but do not execute them'
        )
        
        self.parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        self.parser.add_argument(
            '--no-cache',
            action='store_true',
            help='Disable response caching'
        )
        
        # Advanced options
        self.parser.add_argument(
            '--max-retries',
            type=int,
            default=3,
            help='Maximum number of retries for LLM calls (default: 3)'
        )
        
        self.parser.add_argument(
            '--timeout',
            type=int,
            default=30,
            help='Timeout for LLM calls in seconds (default: 30)'
        )
        
        self.parser.add_argument(
            '--exclude',
            type=str,
            nargs='*',
            default=[],
            help='Additional patterns to exclude from processing'
        )
    
    def parse_args(self) -> TestGenerationConfig:
        """Parse command-line arguments and return configuration."""
        args = self.parser.parse_args()
        
        # Handle interactive mode
        if args.interactive:
            cli = InteractiveCLI()
            if cli.run_interactive_mode():
                cli.display_config()
                return cli.config
            else:
                sys.exit(1)
        
        # Validate repository path
        if not args.repo.exists():
            print(f"Error: Repository path does not exist: {args.repo}")
            sys.exit(1)
        
        # Get API key from environment if not provided
        api_key = args.api_key
        if not api_key and args.model != 'ollama':
            env_vars = {
                'openai': 'OPENAI_API_KEY',
                'gemini': 'GEMINI_API_KEY',
                'claude': 'ANTHROPIC_API_KEY'
            }
            env_var = env_vars.get(args.model)
            if env_var:
                api_key = os.getenv(env_var)
                if not api_key:
                    print(f"Warning: No API key provided for {args.model}. "
                          f"Set {env_var} environment variable or use --api-key")
        
        # Create configuration
        config = TestGenerationConfig(
            target_repo=args.repo,
            llm_provider=LLMProvider(args.model),
            api_key=api_key,
            test_framework=TestFramework(args.framework),
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_cache=not args.no_cache,
            max_retries=args.max_retries,
            timeout=args.timeout,
            exclude_patterns=TestGenerationConfig().exclude_patterns + args.exclude
        )
        
        return config


async def main():
    """Main entry point."""
    try:
        # Parse command-line arguments
        parser = CLIArgumentParser()
        config = parser.parse_args()
        
        # Display configuration in verbose mode
        if config.verbose:
            print("Starting with configuration:")
            print(f"  Repository: {config.target_repo}")
            print(f"  LLM Provider: {config.llm_provider.value}")
            print(f"  Test Framework: {config.test_framework.value}")
            print(f"  Dry Run: {config.dry_run}")
            print(f"  Use Cache: {config.use_cache}")
            print()
        
        # Check LLM backend availability
        backend = LLMFactory.create_backend(config.llm_provider, config.api_key)
        if not backend.is_available():
            print(f"Error: {config.llm_provider.value} backend is not available.")
            print("Please check your API key and required dependencies.")
            sys.exit(1)
        
        # Create and run test generator
        generator = UniversalTestGenerator(config)
        results = await generator.generate_tests_for_repository()
        
        # Display summary
        generator.print_summary()
        
        # Exit with appropriate code
        failed_results = [r for r in results if r.tests_failed > 0]
        if failed_results:
            print(f"\nWarning: {len(failed_results)} modules had test failures.")
            sys.exit(1)
        else:
            print("\nAll tests generated successfully!")
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


def check_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        'requests': 'requests',
        'openai': 'openai',
        'google-generativeai': 'google-generativeai',
        'anthropic': 'anthropic'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("Missing optional dependencies for full functionality:")
        for package in missing_packages:
            print(f"  pip install {package}")
        print("\nNote: You can still use the tool with available backends.")


if __name__ == "__main__":
    # Check dependencies on startup
    check_dependencies()
    
    # Run the main async function
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to start: {str(e)}")
        sys.exit(1)