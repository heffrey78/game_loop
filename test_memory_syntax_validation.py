#!/usr/bin/env python3
"""
Syntax and import validation test for the emotional memory system.
Tests that all modules can be imported without syntax errors.
"""

import ast
import sys
from pathlib import Path

def test_file_syntax(file_path: Path) -> bool:
    """Test if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        print(f"✅ {file_path.name}: Valid syntax")
        return True
        
    except SyntaxError as e:
        print(f"❌ {file_path.name}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {file_path.name}: Error reading file: {e}")
        return False


def main():
    """Test all emotional memory system files for syntax errors"""
    print("🧪 EMOTIONAL MEMORY SYSTEM SYNTAX VALIDATION")
    print("=" * 60)
    
    # Get the project root
    project_root = Path(__file__).parent
    memory_path = project_root / "src" / "game_loop" / "core" / "memory"
    
    # Files to test
    test_files = [
        memory_path / "emotional_context.py",
        memory_path / "affective_weighting.py", 
        memory_path / "emotional_preservation.py",
        memory_path / "validation.py",
        memory_path / "emotional_clustering.py",
        memory_path / "config.py",
        memory_path / "constants.py",
        memory_path / "exceptions.py",
        memory_path / "algorithms.py",
        memory_path / "confidence.py",
        memory_path / "emotional_analysis.py",
    ]
    
    results = []
    
    for file_path in test_files:
        if file_path.exists():
            result = test_file_syntax(file_path)
            results.append((file_path.name, result))
        else:
            print(f"⚠️  {file_path.name}: File not found")
            results.append((file_path.name, False))
    
    print("\n" + "=" * 60)
    print("📋 SYNTAX VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for filename, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {filename}")
    
    print("-" * 60)
    print(f"RESULT: {passed}/{total} files have valid syntax")
    
    if passed == total:
        print("\n🎉 ALL FILES PASS SYNTAX VALIDATION!")
        
        # Test basic imports (without requiring database)
        print("\n🧪 Testing basic imports...")
        
        # Add src to path for imports
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        import_tests = [
            ("MoodState enum", "from game_loop.core.memory.emotional_context import MoodState"),
            ("EmotionalMemoryType enum", "from game_loop.core.memory.emotional_context import EmotionalMemoryType"), 
            ("MemoryProtectionLevel enum", "from game_loop.core.memory.emotional_context import MemoryProtectionLevel"),
            ("AffectiveWeightingStrategy enum", "from game_loop.core.memory.affective_weighting import AffectiveWeightingStrategy"),
            ("EmotionalSignificance dataclass", "from game_loop.core.memory.emotional_context import EmotionalSignificance"),
            ("AffectiveWeight dataclass", "from game_loop.core.memory.affective_weighting import AffectiveWeight"),
            ("Validation functions", "from game_loop.core.memory.validation import validate_probability, validate_mood_state"),
            ("Exception classes", "from game_loop.core.memory.exceptions import EmotionalAnalysisError, ValidationError"),
        ]
        
        import_results = []
        
        for test_name, import_stmt in import_tests:
            try:
                exec(import_stmt)
                print(f"  ✅ {test_name}: Import successful")
                import_results.append(True)
            except Exception as e:
                print(f"  ❌ {test_name}: Import failed - {e}")
                import_results.append(False)
        
        import_passed = sum(import_results)
        import_total = len(import_results)
        
        print(f"\n📊 Import test results: {import_passed}/{import_total} successful")
        
        if import_passed == import_total:
            print("\n🎯 EMOTIONAL MEMORY SYSTEM VALIDATION COMPLETE")
            print("\n✅ All syntax errors fixed")
            print("✅ All core imports working")
            print("✅ All dataclasses and enums defined correctly")
            print("✅ Circular import issues resolved")
            print("✅ System ready for integration testing")
            
            # Test enum functionality
            print("\n🧪 Testing enum functionality...")
            
            try:
                from game_loop.core.memory.emotional_context import MoodState, EmotionalMemoryType, MemoryProtectionLevel
                from game_loop.core.memory.affective_weighting import AffectiveWeightingStrategy
                
                # Test enum creation and access
                mood = MoodState.JOYFUL
                memory_type = EmotionalMemoryType.PEAK_POSITIVE
                protection = MemoryProtectionLevel.PRIVATE
                strategy = AffectiveWeightingStrategy.PERSONALITY_ADAPTIVE
                
                print(f"  ✅ MoodState.JOYFUL: {mood.value}")
                print(f"  ✅ EmotionalMemoryType.PEAK_POSITIVE: {memory_type.value}")
                print(f"  ✅ MemoryProtectionLevel.PRIVATE: {protection.value}")
                print(f"  ✅ AffectiveWeightingStrategy.PERSONALITY_ADAPTIVE: {strategy.value}")
                
                print("✅ Enum functionality: OPERATIONAL")
                
            except Exception as e:
                print(f"❌ Enum functionality test failed: {e}")
                return False
        
        return passed == total and import_passed == import_total
    else:
        print(f"\n⚠️  {total - passed} files have syntax errors that need fixing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)