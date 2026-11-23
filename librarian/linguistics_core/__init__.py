"""Internal modular mixins for LinguisticEngine."""

from .distractor_engine import (
    DistractorEngineMixin,
    DistractorFallbacksMixin,
    DistractorGuardsCollisionMixin,
    DistractorGuardsCountabilityMixin,
    DistractorGuardsInversionMixin,
    DistractorGuardsMixin,
    DistractorGuardsValencyMixin,
    DistractorInflectionsMixin,
    DistractorParonymsMixin,
    DistractorPhrasesMixin,
    DistractorVocabAssemblyMixin,
    DistractorVocabHarvestMixin,
    DistractorVocabMixin,
    DistractorVocabWsdMixin,
)
from .slot_frame import SlotFrame
from .extraction_expressions import (
    ExtractionExpressionsCandidatesMixin,
    ExtractionExpressionsExtractorMixin,
    ExtractionExpressionsMixin,
    ExtractionExpressionsSkeletonsMixin,
)
from .extraction_vocab import (
    ExtractionVocabCandidatesMixin,
    ExtractionVocabExamplesMixin,
    ExtractionVocabExtractorMixin,
    ExtractionVocabMixin,
)
from .quiz_frames import (
    QuizFramesClauseIsolationMixin,
    QuizFramesMixin,
    QuizFramesPatientMixin,
    QuizFramesSlicingMixin,
    QuizFramesSlotSlicingMixin,
)
from .quiz_skeleton_anchors import QuizSkeletonAnchorsMixin
from .quiz_skeleton_builder import QuizSkeletonBuilderMixin
from .quiz_skeleton_context import CXN_ANCHOR_TYPES, SkeletonContext
from .quiz_skeleton_distractors import QuizSkeletonDistractorsMixin
from .quiz_skeleton_frames import QuizSkeletonFramesMixin
from .quiz_skeleton_prep import QuizSkeletonPrepMixin
from .quiz_vocab import QuizVocabMixin
from .quiz_vocab_cloze import QuizVocabClozeMixin
from .quiz_reading import (
    QuizReadingBlueprintsMixin,
    QuizReadingEvaluatorMixin,
    QuizReadingGraphMixin,
    QuizReadingMixin,
)
from .wsd_arbitration import WsdArbitrationMixin
from .wsd_definitions import WsdDefinitionsMixin
from .wsd_engine import WsdEngineMixin
from .wsd_expressions import WsdExpressionsMixin
from .wsd_gates import WsdGatesMixin
from .wsd_lock import WsdLockMixin
from .wsd_scoring import (
    WsdContext,
    WsdScoringDependenciesMixin,
    WsdScoringGatesMixin,
    WsdScoringMixin,
    WsdScoringPatternsMixin,
    WsdSenseInfo,
)
from .wsd_wordnet import WsdWordNetMixin

__all__ = [
    "CXN_ANCHOR_TYPES",
    "DistractorEngineMixin",
    "DistractorFallbacksMixin",
    "DistractorGuardsCollisionMixin",
    "DistractorGuardsCountabilityMixin",
    "DistractorGuardsInversionMixin",
    "DistractorGuardsMixin",
    "DistractorGuardsValencyMixin",
    "DistractorInflectionsMixin",
    "DistractorParonymsMixin",
    "DistractorPhrasesMixin",
    "DistractorVocabAssemblyMixin",
    "DistractorVocabHarvestMixin",
    "DistractorVocabMixin",
    "DistractorVocabWsdMixin",
    "ExtractionExpressionsCandidatesMixin",
    "ExtractionExpressionsExtractorMixin",
    "ExtractionExpressionsMixin",
    "ExtractionExpressionsSkeletonsMixin",
    "ExtractionVocabCandidatesMixin",
    "ExtractionVocabExamplesMixin",
    "ExtractionVocabExtractorMixin",
    "ExtractionVocabMixin",
    "QuizFramesClauseIsolationMixin",
    "QuizFramesMixin",
    "QuizFramesPatientMixin",
    "QuizFramesSlicingMixin",
    "QuizFramesSlotSlicingMixin",
    "QuizSkeletonAnchorsMixin",
    "QuizSkeletonBuilderMixin",
    "QuizSkeletonDistractorsMixin",
    "QuizSkeletonFramesMixin",
    "QuizSkeletonPrepMixin",
    "QuizReadingBlueprintsMixin",
    "QuizReadingEvaluatorMixin",
    "QuizReadingGraphMixin",
    "QuizReadingMixin",
    "QuizVocabClozeMixin",
    "QuizVocabMixin",
    "SkeletonContext",
    "SlotFrame",
    "WsdArbitrationMixin",
    "WsdContext",
    "WsdDefinitionsMixin",
    "WsdEngineMixin",
    "WsdExpressionsMixin",
    "WsdGatesMixin",
    "WsdLockMixin",
    "WsdScoringDependenciesMixin",
    "WsdScoringGatesMixin",
    "WsdScoringMixin",
    "WsdScoringPatternsMixin",
    "WsdSenseInfo",
    "WsdWordNetMixin",
]
