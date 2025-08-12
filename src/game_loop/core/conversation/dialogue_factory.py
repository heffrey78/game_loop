"""Factory for creating dialogue-aware conversation flow managers with memory integration."""

import uuid
from typing import Optional

from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient

from .context_engine import DialogueMemoryIntegrationEngine
from .conversation_threading import ConversationThreadingService
from .flow_manager import EnhancedConversationFlowManager
from .memory_integration import MemoryIntegrationInterface


class DialogueAwareFlowManagerFactory:
    """Factory for creating conversation flow managers with advanced memory integration."""

    @staticmethod
    def create_enhanced_flow_manager(
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        memory_integration: MemoryIntegrationInterface,
        threading_service: Optional[ConversationThreadingService] = None,
        enable_dialogue_integration: bool = True,
        naturalness_threshold: float = 0.7,
        engagement_threshold: float = 0.6,
    ) -> EnhancedConversationFlowManager:
        """Create an enhanced conversation flow manager with dialogue-aware memory integration.

        Args:
            session_factory: Database session factory
            llm_client: LLM client for natural language processing
            memory_integration: Memory integration interface
            threading_service: Optional conversation threading service
            enable_dialogue_integration: Whether to enable dialogue memory integration
            naturalness_threshold: Minimum naturalness score for memory integration
            engagement_threshold: Minimum engagement level for memory integration

        Returns:
            Enhanced conversation flow manager with dialogue integration
        """

        # Create threading service if not provided
        if threading_service is None:
            threading_service = ConversationThreadingService(
                session_factory=session_factory,
                memory_integration=memory_integration,
                enable_threading=True,
            )

        # Create dialogue memory integration engine
        dialogue_engine = None
        if enable_dialogue_integration:
            dialogue_engine = DialogueMemoryIntegrationEngine(
                session_factory=session_factory,
                llm_client=llm_client,
                threading_service=threading_service,
                memory_integration=memory_integration,
                naturalness_threshold=naturalness_threshold,
                engagement_threshold=engagement_threshold,
            )

        # Create enhanced flow manager
        flow_manager = EnhancedConversationFlowManager(
            memory_integration=memory_integration,
            session_factory=session_factory,
            enable_conversation_threading=True,
            enable_dialogue_integration=enable_dialogue_integration,
            dialogue_integration_engine=dialogue_engine,
        )

        return flow_manager

    @staticmethod
    def create_standard_flow_manager(
        session_factory: DatabaseSessionFactory,
        memory_integration: MemoryIntegrationInterface,
    ) -> EnhancedConversationFlowManager:
        """Create a standard flow manager without advanced dialogue integration.

        Args:
            session_factory: Database session factory
            memory_integration: Memory integration interface

        Returns:
            Standard conversation flow manager
        """
        return EnhancedConversationFlowManager(
            memory_integration=memory_integration,
            session_factory=session_factory,
            enable_conversation_threading=False,
            enable_dialogue_integration=False,
        )

    @staticmethod
    def create_threading_only_flow_manager(
        session_factory: DatabaseSessionFactory,
        memory_integration: MemoryIntegrationInterface,
        threading_service: Optional[ConversationThreadingService] = None,
    ) -> EnhancedConversationFlowManager:
        """Create a flow manager with only conversation threading (no dialogue integration).

        Args:
            session_factory: Database session factory
            memory_integration: Memory integration interface
            threading_service: Optional conversation threading service

        Returns:
            Flow manager with threading but no dialogue integration
        """
        if threading_service is None:
            threading_service = ConversationThreadingService(
                session_factory=session_factory,
                memory_integration=memory_integration,
                enable_threading=True,
            )

        return EnhancedConversationFlowManager(
            memory_integration=memory_integration,
            session_factory=session_factory,
            enable_conversation_threading=True,
            enable_dialogue_integration=False,
            threading_service=threading_service,
        )

    @staticmethod
    def create_dialogue_only_flow_manager(
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        memory_integration: MemoryIntegrationInterface,
        naturalness_threshold: float = 0.7,
        engagement_threshold: float = 0.6,
    ) -> EnhancedConversationFlowManager:
        """Create a flow manager with only dialogue integration (no threading).

        Args:
            session_factory: Database session factory
            llm_client: LLM client for natural language processing
            memory_integration: Memory integration interface
            naturalness_threshold: Minimum naturalness score for memory integration
            engagement_threshold: Minimum engagement level for memory integration

        Returns:
            Flow manager with dialogue integration but no threading
        """
        # Create mock threading service (disabled)
        threading_service = ConversationThreadingService(
            session_factory=session_factory,
            memory_integration=memory_integration,
            enable_threading=False,
        )

        dialogue_engine = DialogueMemoryIntegrationEngine(
            session_factory=session_factory,
            llm_client=llm_client,
            threading_service=threading_service,
            memory_integration=memory_integration,
            naturalness_threshold=naturalness_threshold,
            engagement_threshold=engagement_threshold,
        )

        return EnhancedConversationFlowManager(
            memory_integration=memory_integration,
            session_factory=session_factory,
            enable_conversation_threading=False,
            enable_dialogue_integration=True,
            dialogue_integration_engine=dialogue_engine,
        )

    @staticmethod
    def create_for_testing(
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        memory_integration: MemoryIntegrationInterface,
        disable_all_enhancements: bool = False,
    ) -> EnhancedConversationFlowManager:
        """Create a flow manager optimized for testing.

        Args:
            session_factory: Database session factory
            llm_client: LLM client for natural language processing
            memory_integration: Memory integration interface
            disable_all_enhancements: Whether to disable all enhancements for clean testing

        Returns:
            Flow manager configured for testing
        """
        if disable_all_enhancements:
            return DialogueAwareFlowManagerFactory.create_standard_flow_manager(
                session_factory=session_factory,
                memory_integration=memory_integration,
            )

        # Create enhanced flow manager with lower thresholds for testing
        return DialogueAwareFlowManagerFactory.create_enhanced_flow_manager(
            session_factory=session_factory,
            llm_client=llm_client,
            memory_integration=memory_integration,
            naturalness_threshold=0.5,  # Lower threshold for testing
            engagement_threshold=0.4,  # Lower threshold for testing
        )
