"""Tests for interface compliance and segregation in conversation system."""

import inspect
import uuid
from abc import ABC
from unittest.mock import AsyncMock, Mock

import pytest

from game_loop.core.conversation.interfaces import ConversationMemoryInterface
from game_loop.core.conversation.memory_integration import (
    MemoryContext,
    MemoryIntegrationInterface,
    MemoryRetrievalResult,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
)
from game_loop.core.conversation.flow_manager import ConversationFlowManager


class TestInterfaceSegregation:
    """Test that interfaces follow the Interface Segregation Principle."""

    def test_conversation_memory_interface_is_abstract(self):
        """Test that ConversationMemoryInterface is properly abstract."""
        assert issubclass(ConversationMemoryInterface, ABC)

        # Should not be able to instantiate abstract interface
        with pytest.raises(TypeError):
            ConversationMemoryInterface()

    def test_conversation_memory_interface_has_minimal_methods(self):
        """Test that the interface only has the methods actually needed."""
        # Get abstract methods
        abstract_methods = ConversationMemoryInterface.__abstractmethods__

        # Should only have the two methods used by ConversationFlowManager
        expected_methods = {"extract_memory_context", "retrieve_relevant_memories"}
        assert abstract_methods == expected_methods

    def test_memory_integration_interface_implements_abstract_interface(self):
        """Test that MemoryIntegrationInterface properly implements the abstract interface."""
        assert issubclass(MemoryIntegrationInterface, ConversationMemoryInterface)

        # Should be able to instantiate the concrete implementation
        mock_session_factory = Mock()
        mock_llm_client = Mock()

        implementation = MemoryIntegrationInterface(
            session_factory=mock_session_factory,
            llm_client=mock_llm_client,
        )

        assert isinstance(implementation, ConversationMemoryInterface)

    def test_interface_method_signatures_match(self):
        """Test that implementation method signatures match the interface."""
        # Get interface methods
        interface_methods = {}
        for name, method in inspect.getmembers(
            ConversationMemoryInterface, predicate=inspect.isfunction
        ):
            if not name.startswith("_"):
                interface_methods[name] = inspect.signature(method)

        # Get implementation methods
        implementation_methods = {}
        for name, method in inspect.getmembers(
            MemoryIntegrationInterface, predicate=inspect.ismethod
        ):
            if name in interface_methods:
                implementation_methods[name] = inspect.signature(method)

        # Compare signatures (excluding 'self' parameter)
        for method_name in interface_methods:
            if method_name in implementation_methods:
                interface_params = list(
                    interface_methods[method_name].parameters.values()
                )[
                    1:
                ]  # Skip 'self'
                impl_params = list(
                    implementation_methods[method_name].parameters.values()
                )[
                    1:
                ]  # Skip 'self'

                assert len(interface_params) == len(
                    impl_params
                ), f"Parameter count mismatch in {method_name}"

                for interface_param, impl_param in zip(interface_params, impl_params):
                    assert interface_param.name == impl_param.name, (
                        f"Parameter name mismatch in {method_name}: "
                        f"{interface_param.name} vs {impl_param.name}"
                    )
                    assert interface_param.annotation == impl_param.annotation, (
                        f"Parameter type mismatch in {method_name}.{interface_param.name}: "
                        f"{interface_param.annotation} vs {impl_param.annotation}"
                    )


class TestConversationFlowManagerInterfaceDependency:
    """Test that ConversationFlowManager properly depends on the interface."""

    def test_flow_manager_accepts_interface(self):
        """Test that ConversationFlowManager accepts the interface type."""
        # Create a mock that implements the interface
        mock_memory_integration = Mock(spec=ConversationMemoryInterface)
        mock_session_factory = Mock()

        # Should be able to create flow manager with interface
        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
        )

        assert flow_manager.memory_integration is mock_memory_integration

    def test_flow_manager_only_uses_interface_methods(self):
        """Test that ConversationFlowManager only calls methods defined in the interface."""
        # Get all methods called on memory_integration in flow_manager.py
        import ast
        import inspect

        # Get the source code of ConversationFlowManager
        flow_manager_source = inspect.getsource(ConversationFlowManager)

        # Parse the AST to find method calls on self.memory_integration
        class MethodCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.memory_integration_calls = set()

            def visit_Attribute(self, node):
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "self"
                    and node.value.attr == "memory_integration"
                ):
                    self.memory_integration_calls.add(node.attr)
                self.generic_visit(node)

        tree = ast.parse(flow_manager_source)
        visitor = MethodCallVisitor()
        visitor.visit(tree)

        # All calls should be to interface methods
        interface_methods = set(ConversationMemoryInterface.__abstractmethods__)

        # Verify that all calls are to interface methods
        unauthorized_calls = visitor.memory_integration_calls - interface_methods
        assert (
            not unauthorized_calls
        ), f"ConversationFlowManager calls methods not in the interface: {unauthorized_calls}"


class TestMockConversationMemoryInterface:
    """Test that we can create mocks of the interface for testing."""

    @pytest.mark.asyncio
    async def test_mock_interface_for_testing(self):
        """Test that the interface can be easily mocked for testing."""
        # Create a mock implementation
        mock_memory_integration = AsyncMock(spec=ConversationMemoryInterface)

        # Configure mock returns
        mock_context = MemoryContext()
        mock_result = MemoryRetrievalResult()

        mock_memory_integration.extract_memory_context.return_value = mock_context
        mock_memory_integration.retrieve_relevant_memories.return_value = mock_result

        # Test that the mock works as expected
        conversation = Mock(spec=ConversationContext)
        personality = Mock(spec=NPCPersonality)

        context = await mock_memory_integration.extract_memory_context(
            conversation, "test message", personality
        )
        assert context is mock_context

        memories = await mock_memory_integration.retrieve_relevant_memories(
            mock_context, uuid.uuid4()
        )
        assert memories is mock_result

        # Verify calls were made
        mock_memory_integration.extract_memory_context.assert_called_once()
        mock_memory_integration.retrieve_relevant_memories.assert_called_once()


class TestInterfaceDocumentation:
    """Test that interfaces have proper documentation."""

    def test_interface_has_docstring(self):
        """Test that the interface has proper documentation."""
        assert ConversationMemoryInterface.__doc__ is not None
        assert len(ConversationMemoryInterface.__doc__.strip()) > 0

        # Check that it mentions Interface Segregation Principle
        assert "Interface Segregation Principle" in ConversationMemoryInterface.__doc__

    def test_interface_methods_have_docstrings(self):
        """Test that interface methods have proper documentation."""
        for name in ConversationMemoryInterface.__abstractmethods__:
            method = getattr(ConversationMemoryInterface, name)
            assert method.__doc__ is not None, f"Method {name} lacks documentation"
            assert (
                len(method.__doc__.strip()) > 0
            ), f"Method {name} has empty documentation"

            # Should document parameters and return values
            doc = method.__doc__
            assert "Args:" in doc, f"Method {name} doesn't document parameters"
            assert "Returns:" in doc, f"Method {name} doesn't document return value"


class TestInterfaceBenefits:
    """Test that the interface provides the expected benefits."""

    def test_interface_enables_dependency_injection(self):
        """Test that the interface enables proper dependency injection."""
        # Can inject different implementations
        mock_impl_1 = Mock(spec=ConversationMemoryInterface)
        mock_impl_2 = Mock(spec=ConversationMemoryInterface)

        mock_session_factory = Mock()

        flow_manager_1 = ConversationFlowManager(mock_impl_1, mock_session_factory)
        flow_manager_2 = ConversationFlowManager(mock_impl_2, mock_session_factory)

        assert flow_manager_1.memory_integration is mock_impl_1
        assert flow_manager_2.memory_integration is mock_impl_2
        assert (
            flow_manager_1.memory_integration is not flow_manager_2.memory_integration
        )

    def test_interface_enables_testing_isolation(self):
        """Test that the interface enables isolated unit testing."""
        # Can create ConversationFlowManager without concrete dependencies
        mock_memory_integration = Mock(spec=ConversationMemoryInterface)
        mock_session_factory = Mock()

        flow_manager = ConversationFlowManager(
            memory_integration=mock_memory_integration,
            session_factory=mock_session_factory,
        )

        # ConversationFlowManager doesn't know about concrete implementation details
        assert not hasattr(flow_manager.memory_integration, "llm_client")
        assert not hasattr(flow_manager.memory_integration, "similarity_threshold")

        # Only knows about interface methods
        assert hasattr(flow_manager.memory_integration, "extract_memory_context")
        assert hasattr(flow_manager.memory_integration, "retrieve_relevant_memories")
