"""
Generation Quality Monitor for Dynamic World Integration.

Monitors the quality of generated content and provides feedback for 
continuous improvement.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Any
from uuid import UUID

from game_loop.state.models import (
    GeneratedContent,
    PerformanceMetrics,
    QualityAssessment,
    QualityImprovement,
    QualityIssue,
    QualityReport,
    QualityTrends,
    SatisfactionData,
)

logger = logging.getLogger(__name__)


class GenerationQualityMonitor:
    """
    Monitors and assesses the quality of generated content.
    
    This class provides:
    - Quality assessment for generated content
    - Performance monitoring for generation systems
    - Issue detection and quality improvement suggestions
    - Trend analysis and reporting
    """

    def __init__(self, session_factory):
        """Initialize quality monitoring system."""
        self.session_factory = session_factory
        self.quality_cache = {}
        self.performance_cache = {}
        
        # Quality assessment weights
        self.quality_weights = {
            "consistency": 0.25,
            "creativity": 0.20,
            "relevance": 0.25,
            "engagement": 0.20,
            "technical": 0.10,
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "poor": 0.40,
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "generation_time_ms": 5000,     # 5 seconds max
            "memory_usage_mb": 512,         # 512 MB max
            "cpu_usage_percent": 80,        # 80% max
        }

    async def assess_content_quality(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> QualityAssessment:
        """
        Assess quality of generated content.
        
        Args:
            content: The generated content to assess
            context: Additional context for assessment
            
        Returns:
            QualityAssessment with detailed quality metrics
        """
        try:
            assessment = QualityAssessment(
                content_id=content.content_id,
                overall_quality_score=0.0,
                assessment_method="automated",
            )
            
            # Assess different quality dimensions
            dimension_scores = {}
            
            # Consistency assessment
            consistency_score = await self._assess_consistency(content, context)
            dimension_scores["consistency"] = consistency_score
            
            # Creativity assessment
            creativity_score = await self._assess_creativity(content, context)
            dimension_scores["creativity"] = creativity_score
            
            # Relevance assessment
            relevance_score = await self._assess_relevance(content, context)
            dimension_scores["relevance"] = relevance_score
            
            # Engagement assessment
            engagement_score = await self._assess_engagement(content, context)
            dimension_scores["engagement"] = engagement_score
            
            # Technical quality assessment
            technical_score = await self._assess_technical_quality(content, context)
            dimension_scores["technical"] = technical_score
            
            assessment.dimension_scores = dimension_scores
            
            # Calculate overall quality score
            overall_score = sum(
                score * self.quality_weights.get(dimension, 0.2)
                for dimension, score in dimension_scores.items()
            )
            assessment.overall_quality_score = max(0.0, min(1.0, overall_score))
            
            # Generate improvement suggestions
            assessment.improvement_suggestions = await self._generate_improvement_suggestions(
                dimension_scores, content
            )
            
            # Store assessment in database
            await self._store_quality_assessment(assessment)
            
            # Cache assessment
            self.quality_cache[content.content_id] = assessment
            
            logger.info(f"Assessed quality for content {content.content_id}: {assessment.overall_quality_score:.2f}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            return QualityAssessment(
                content_id=content.content_id,
                overall_quality_score=0.5,  # Default score
                assessment_method="error_fallback",
            )

    async def track_player_satisfaction(
        self, content_id: UUID, satisfaction_data: SatisfactionData
    ) -> bool:
        """
        Track player satisfaction with generated content.
        
        Args:
            content_id: ID of the content
            satisfaction_data: Player satisfaction data
            
        Returns:
            True if tracking was successful
        """
        try:
            # Validate satisfaction data
            if not self._validate_satisfaction_data(satisfaction_data):
                logger.warning(f"Invalid satisfaction data for content {content_id}")
                return False
            
            # Store satisfaction data
            await self._store_satisfaction_data(content_id, satisfaction_data)
            
            # Update quality assessment if needed
            if content_id in self.quality_cache:
                await self._update_quality_with_satisfaction(content_id, satisfaction_data)
            
            # Check if satisfaction indicates quality issues
            if satisfaction_data.rating <= 2:
                await self._flag_potential_quality_issue(content_id, satisfaction_data)
            
            logger.info(f"Tracked satisfaction for content {content_id}: {satisfaction_data.rating}/5")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking player satisfaction: {e}")
            return False

    async def monitor_generation_performance(
        self, generation_event: dict[str, Any]
    ) -> PerformanceMetrics:
        """
        Monitor performance metrics of content generation.
        
        Args:
            generation_event: Event data from content generation
            
        Returns:
            PerformanceMetrics with measured performance data
        """
        try:
            metrics = PerformanceMetrics()
            
            # Extract performance data from event
            metrics.generation_time = generation_event.get("generation_time", 0.0)
            metrics.memory_usage = generation_event.get("memory_usage", 0.0)
            metrics.cpu_usage = generation_event.get("cpu_usage", 0.0)
            metrics.database_queries = generation_event.get("database_queries", 0)
            
            # Calculate cache hit rate if available
            cache_hits = generation_event.get("cache_hits", 0)
            cache_misses = generation_event.get("cache_misses", 0)
            total_cache_requests = cache_hits + cache_misses
            if total_cache_requests > 0:
                metrics.cache_hit_rate = cache_hits / total_cache_requests
            
            # Store performance metrics
            await self._store_performance_metrics(metrics, generation_event)
            
            # Check for performance issues
            await self._check_performance_thresholds(metrics, generation_event)
            
            # Cache metrics
            content_id = generation_event.get("content_id")
            if content_id:
                self.performance_cache[content_id] = metrics
            
            logger.info(f"Monitored performance: {metrics.generation_time:.2f}ms generation time")
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring generation performance: {e}")
            return PerformanceMetrics()

    async def detect_quality_issues(
        self, content_batch: list[GeneratedContent]
    ) -> list[QualityIssue]:
        """
        Detect potential quality issues in generated content.
        
        Args:
            content_batch: Batch of generated content to analyze
            
        Returns:
            List of detected quality issues
        """
        try:
            issues = []
            
            # Analyze each piece of content
            for content in content_batch:
                content_issues = await self._analyze_content_for_issues(content)
                issues.extend(content_issues)
            
            # Analyze batch-level issues
            batch_issues = await self._analyze_batch_consistency(content_batch)
            issues.extend(batch_issues)
            
            # Prioritize issues by severity
            issues.sort(key=lambda x: self._severity_priority(x.severity), reverse=True)
            
            # Store issues in database
            for issue in issues:
                await self._store_quality_issue(issue)
            
            logger.info(f"Detected {len(issues)} quality issues in batch of {len(content_batch)} items")
            return issues
            
        except Exception as e:
            logger.error(f"Error detecting quality issues: {e}")
            return []

    async def get_quality_trends(
        self, content_type: str, timeframe_days: int = 30
    ) -> QualityTrends:
        """
        Analyze quality trends over time.
        
        Args:
            content_type: Type of content to analyze
            timeframe_days: Number of days to analyze
            
        Returns:
            QualityTrends with trend analysis
        """
        try:
            trends = QualityTrends()
            
            # Get quality assessments over time
            assessments = await self._get_quality_assessments_over_time(
                content_type, timeframe_days
            )
            
            if assessments:
                # Extract quality scores with timestamps
                quality_data = [
                    (assessment.assessed_at, assessment.overall_quality_score)
                    for assessment in assessments
                ]
                trends.quality_scores_over_time = quality_data
                
                # Calculate improvement rate
                if len(quality_data) > 1:
                    scores = [score for _, score in quality_data]
                    trends.improvement_rate = await self._calculate_improvement_rate(scores)
                
                # Determine trend direction
                trends.trend_direction = await self._determine_trend_direction(quality_data)
                
                # Identify key factors affecting quality
                trends.key_factors = await self._identify_quality_factors(
                    content_type, assessments
                )
            
            logger.info(f"Analyzed quality trends for {content_type}")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing quality trends: {e}")
            return QualityTrends()

    async def generate_quality_report(self, timeframe_days: int = 7) -> QualityReport:
        """
        Generate comprehensive quality report.
        
        Args:
            timeframe_days: Number of days to include in report
            
        Returns:
            QualityReport with comprehensive quality analysis
        """
        try:
            start_date = datetime.now() - timedelta(days=timeframe_days)
            end_date = datetime.now()
            
            report = QualityReport(
                report_period=(start_date, end_date),
                overall_quality_score=0.0,
            )
            
            # Get quality data for the period
            quality_data = await self._get_quality_data_for_period(start_date, end_date)
            
            if quality_data:
                # Calculate overall quality score
                all_scores = [item["quality_score"] for item in quality_data]
                report.overall_quality_score = mean(all_scores)
                
                # Calculate quality by content type
                quality_by_type = defaultdict(list)
                for item in quality_data:
                    quality_by_type[item["content_type"]].append(item["quality_score"])
                
                report.quality_by_content_type = {
                    content_type: mean(scores)
                    for content_type, scores in quality_by_type.items()
                }
                
                # Get major issues
                report.major_issues = await self._get_major_issues_for_period(
                    start_date, end_date
                )
                
                # Get improvements made
                report.improvements_made = await self._get_improvements_for_period(
                    start_date, end_date
                )
                
                # Generate recommendations
                report.recommendations = await self._generate_quality_recommendations(
                    quality_data, report.major_issues
                )
            
            logger.info(f"Generated quality report for {timeframe_days} days")
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return QualityReport(
                report_period=(datetime.now(), datetime.now()),
                overall_quality_score=0.5,
            )

    async def suggest_quality_improvements(
        self, quality_issues: list[QualityIssue]
    ) -> list[QualityImprovement]:
        """
        Suggest improvements based on quality analysis.
        
        Args:
            quality_issues: List of quality issues to address
            
        Returns:
            List of quality improvement suggestions
        """
        try:
            improvements = []
            
            # Group issues by type and system
            issue_groups = self._group_issues(quality_issues)
            
            # Generate improvements for each group
            for group_key, group_issues in issue_groups.items():
                improvement = await self._create_improvement_for_group(
                    group_key, group_issues
                )
                if improvement:
                    improvements.append(improvement)
            
            # Sort by expected impact
            improvements.sort(key=lambda x: x.expected_impact, reverse=True)
            
            # Store improvement suggestions
            for improvement in improvements:
                await self._store_quality_improvement(improvement)
            
            logger.info(f"Generated {len(improvements)} quality improvement suggestions")
            return improvements
            
        except Exception as e:
            logger.error(f"Error suggesting quality improvements: {e}")
            return []

    # Private helper methods

    async def _assess_consistency(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> float:
        """Assess consistency of generated content."""
        # Evaluate consistency with existing content and world rules
        consistency_score = 0.7  # Default good consistency
        
        # Check theme consistency
        if "world_theme" in context and "content_theme" in content.generation_metadata:
            if context["world_theme"] == content.generation_metadata["content_theme"]:
                consistency_score += 0.1
            else:
                consistency_score -= 0.2
        
        # Check narrative consistency
        if "narrative_context" in context:
            # Evaluate how well content fits narrative context
            narrative_fit = await self._evaluate_narrative_fit(content, context)
            consistency_score = (consistency_score + narrative_fit) / 2
        
        return max(0.0, min(1.0, consistency_score))

    async def _assess_creativity(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> float:
        """Assess creativity and originality of content."""
        # Evaluate uniqueness and creative elements
        creativity_score = 0.6  # Default moderate creativity
        
        # Check for unique elements
        if "unique_features" in content.generation_metadata:
            unique_count = len(content.generation_metadata["unique_features"])
            creativity_score += min(0.3, unique_count * 0.1)
        
        # Check for creative descriptions
        description = content.content_data.get("description", "")
        if len(description) > 100:  # Detailed descriptions indicate creativity
            creativity_score += 0.1
        
        # Check for unexpected elements
        if "surprising_elements" in content.generation_metadata:
            creativity_score += 0.2
        
        return max(0.0, min(1.0, creativity_score))

    async def _assess_relevance(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> float:
        """Assess relevance to player and context."""
        # Evaluate how relevant content is to current context
        relevance_score = 0.7  # Default good relevance
        
        # Check player preferences alignment
        if "player_preferences" in context:
            prefs = context["player_preferences"]
            content_type = content.content_type
            if content_type in prefs and prefs[content_type] > 0.6:
                relevance_score += 0.2
            elif content_type in prefs and prefs[content_type] < 0.4:
                relevance_score -= 0.2
        
        # Check contextual appropriateness
        if "generation_purpose" in content.generation_metadata:
            purpose = content.generation_metadata["generation_purpose"]
            if purpose in ["quest_path", "exploration"]:
                relevance_score += 0.1  # These purposes indicate high relevance
        
        return max(0.0, min(1.0, relevance_score))

    async def _assess_engagement(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> float:
        """Assess potential for player engagement."""
        # Evaluate engagement potential
        engagement_score = 0.6  # Default moderate engagement
        
        # Check for interactive elements
        if "interactive_features" in content.content_data:
            interaction_count = len(content.content_data["interactive_features"])
            engagement_score += min(0.3, interaction_count * 0.1)
        
        # Check for compelling descriptions
        description = content.content_data.get("description", "")
        if "mysterious" in description.lower() or "intriguing" in description.lower():
            engagement_score += 0.1
        
        # Check for rewards or benefits
        if "rewards" in content.content_data or "benefits" in content.content_data:
            engagement_score += 0.2
        
        return max(0.0, min(1.0, engagement_score))

    async def _assess_technical_quality(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> float:
        """Assess technical quality of content."""
        # Evaluate technical correctness and completeness
        technical_score = 0.8  # Default good technical quality
        
        # Check for required fields
        required_fields = ["name", "description"]
        missing_fields = [
            field for field in required_fields
            if field not in content.content_data or not content.content_data[field]
        ]
        if missing_fields:
            technical_score -= len(missing_fields) * 0.2
        
        # Check for data validation
        if "validation_errors" in content.generation_metadata:
            error_count = len(content.generation_metadata["validation_errors"])
            technical_score -= min(0.5, error_count * 0.1)
        
        # Check generation time (faster = better technical implementation)
        generation_time = content.generation_metadata.get("generation_time", 0)
        if generation_time > 5000:  # More than 5 seconds
            technical_score -= 0.1
        
        return max(0.0, min(1.0, technical_score))

    async def _generate_improvement_suggestions(
        self, dimension_scores: dict[str, float], content: GeneratedContent
    ) -> list[str]:
        """Generate improvement suggestions based on dimension scores."""
        suggestions = []
        
        for dimension, score in dimension_scores.items():
            if score < 0.6:  # Below acceptable threshold
                if dimension == "consistency":
                    suggestions.append("Improve consistency with world theme and narrative context")
                elif dimension == "creativity":
                    suggestions.append("Add more unique and creative elements")
                elif dimension == "relevance":
                    suggestions.append("Better align content with player preferences and context")
                elif dimension == "engagement":
                    suggestions.append("Include more interactive and compelling features")
                elif dimension == "technical":
                    suggestions.append("Fix technical issues and improve completeness")
        
        return suggestions

    def _validate_satisfaction_data(self, satisfaction_data: SatisfactionData) -> bool:
        """Validate satisfaction data."""
        return (
            1 <= satisfaction_data.rating <= 5 and
            satisfaction_data.completion_status in ["completed", "abandoned", "incomplete"]
        )

    async def _store_quality_assessment(self, assessment: QualityAssessment) -> None:
        """Store quality assessment in database."""
        # This would insert into generation_quality_metrics table
        pass

    async def _store_satisfaction_data(
        self, content_id: UUID, satisfaction_data: SatisfactionData
    ) -> None:
        """Store satisfaction data in database."""
        # This would update content interaction records
        pass

    async def _update_quality_with_satisfaction(
        self, content_id: UUID, satisfaction_data: SatisfactionData
    ) -> None:
        """Update quality assessment with satisfaction data."""
        # This would update the cached quality assessment
        pass

    async def _flag_potential_quality_issue(
        self, content_id: UUID, satisfaction_data: SatisfactionData
    ) -> None:
        """Flag potential quality issue based on low satisfaction."""
        issue = QualityIssue(
            content_id=content_id,
            issue_type="engagement",
            severity="medium" if satisfaction_data.rating == 2 else "high",
            description=f"Low player satisfaction: {satisfaction_data.rating}/5",
            suggested_fix="Review content engagement and relevance",
        )
        await self._store_quality_issue(issue)

    async def _store_performance_metrics(
        self, metrics: PerformanceMetrics, generation_event: dict[str, Any]
    ) -> None:
        """Store performance metrics in database."""
        # This would insert into generation_performance_logs table
        pass

    async def _check_performance_thresholds(
        self, metrics: PerformanceMetrics, generation_event: dict[str, Any]
    ) -> None:
        """Check if performance metrics exceed thresholds."""
        if metrics.generation_time > self.performance_thresholds["generation_time_ms"]:
            await self._flag_performance_issue("generation_time", metrics, generation_event)
        
        if metrics.memory_usage > self.performance_thresholds["memory_usage_mb"]:
            await self._flag_performance_issue("memory_usage", metrics, generation_event)
        
        if metrics.cpu_usage > self.performance_thresholds["cpu_usage_percent"]:
            await self._flag_performance_issue("cpu_usage", metrics, generation_event)

    async def _flag_performance_issue(
        self, metric_type: str, metrics: PerformanceMetrics, generation_event: dict[str, Any]
    ) -> None:
        """Flag performance issue."""
        content_id = generation_event.get("content_id", "unknown")
        issue = QualityIssue(
            content_id=content_id,
            issue_type="performance",
            severity="medium",
            description=f"Performance threshold exceeded for {metric_type}",
            suggested_fix=f"Optimize {metric_type} in generation system",
        )
        await self._store_quality_issue(issue)

    async def _analyze_content_for_issues(self, content: GeneratedContent) -> list[QualityIssue]:
        """Analyze individual content for quality issues."""
        issues = []
        
        # Check for missing required data
        if not content.content_data.get("name"):
            issues.append(QualityIssue(
                content_id=content.content_id,
                issue_type="technical",
                severity="high",
                description="Missing required name field",
                suggested_fix="Ensure name generation is working correctly",
            ))
        
        # Check for very short descriptions
        description = content.content_data.get("description", "")
        if len(description) < 20:
            issues.append(QualityIssue(
                content_id=content.content_id,
                issue_type="quality",
                severity="medium",
                description="Description too short",
                suggested_fix="Generate more detailed descriptions",
            ))
        
        return issues

    async def _analyze_batch_consistency(
        self, content_batch: list[GeneratedContent]
    ) -> list[QualityIssue]:
        """Analyze consistency across a batch of content."""
        issues = []
        
        # Check for theme consistency
        themes = [
            content.generation_metadata.get("theme")
            for content in content_batch
            if content.generation_metadata.get("theme")
        ]
        
        if len(set(themes)) > len(themes) * 0.7:  # Too many different themes
            issues.append(QualityIssue(
                content_id=content_batch[0].content_id if content_batch else "batch",
                issue_type="consistency",
                severity="medium",
                description="Batch contains too many different themes",
                suggested_fix="Improve theme consistency in batch generation",
            ))
        
        return issues

    def _severity_priority(self, severity: str) -> int:
        """Convert severity to priority number."""
        priorities = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return priorities.get(severity, 1)

    async def _store_quality_issue(self, issue: QualityIssue) -> None:
        """Store quality issue in database."""
        # This would insert into a quality_issues table
        pass

    async def _get_quality_assessments_over_time(
        self, content_type: str, timeframe_days: int
    ) -> list[QualityAssessment]:
        """Get quality assessments over time."""
        # Mock implementation - would query database
        return []

    async def _calculate_improvement_rate(self, scores: list[float]) -> float:
        """Calculate quality improvement rate."""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear improvement rate
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        if first_half and second_half:
            return mean(second_half) - mean(first_half)
        return 0.0

    async def _determine_trend_direction(
        self, quality_data: list[tuple[datetime, float]]
    ) -> str:
        """Determine overall trend direction."""
        if len(quality_data) < 3:
            return "stable"
        
        scores = [score for _, score in quality_data]
        recent_scores = scores[-3:]
        
        if mean(recent_scores) > mean(scores[:-3]):
            return "improving"
        elif mean(recent_scores) < mean(scores[:-3]):
            return "declining"
        else:
            return "stable"

    async def _identify_quality_factors(
        self, content_type: str, assessments: list[QualityAssessment]
    ) -> list[str]:
        """Identify key factors affecting quality."""
        # Mock implementation - would analyze correlations
        return ["generation_time", "player_feedback", "template_quality"]

    async def _get_quality_data_for_period(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Get quality data for a specific period."""
        # Mock implementation - would query database
        return []

    async def _get_major_issues_for_period(
        self, start_date: datetime, end_date: datetime
    ) -> list[QualityIssue]:
        """Get major quality issues for a period."""
        # Mock implementation - would query database
        return []

    async def _get_improvements_for_period(
        self, start_date: datetime, end_date: datetime
    ) -> list[str]:
        """Get improvements made during a period."""
        # Mock implementation - would query database
        return []

    async def _generate_quality_recommendations(
        self, quality_data: list[dict[str, Any]], major_issues: list[QualityIssue]
    ) -> list[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if major_issues:
            recommendations.append("Address high-severity quality issues first")
        
        # Add more recommendations based on data analysis
        recommendations.extend([
            "Monitor generation performance regularly",
            "Collect more player feedback for quality assessment",
            "Review and update generation templates",
        ])
        
        return recommendations

    def _group_issues(self, quality_issues: list[QualityIssue]) -> dict[str, list[QualityIssue]]:
        """Group quality issues by type and system."""
        groups = defaultdict(list)
        
        for issue in quality_issues:
            group_key = f"{issue.issue_type}_{issue.severity}"
            groups[group_key].append(issue)
        
        return groups

    async def _create_improvement_for_group(
        self, group_key: str, group_issues: list[QualityIssue]
    ) -> QualityImprovement | None:
        """Create improvement suggestion for a group of issues."""
        if not group_issues:
            return None
        
        issue_type, severity = group_key.split("_", 1)
        
        improvement = QualityImprovement(
            target_issue_types=[issue_type],
            improvement_type="parameter_adjustment",
            description=f"Address {len(group_issues)} {severity} {issue_type} issues",
            expected_impact=0.1 * len(group_issues),  # More issues = higher impact
            implementation_difficulty="medium",
        )
        
        return improvement

    async def _store_quality_improvement(self, improvement: QualityImprovement) -> None:
        """Store quality improvement suggestion in database."""
        # This would insert into quality_improvements table
        pass

    async def _evaluate_narrative_fit(
        self, content: GeneratedContent, context: dict[str, Any]
    ) -> float:
        """Evaluate how well content fits narrative context."""
        # Mock implementation - would analyze narrative consistency
        return 0.7