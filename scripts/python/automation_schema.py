#!/usr/bin/env python3
"""
Automation Timeline Schema for Beat-Synchronized Control

This module defines the JSON schema for automation events that control
Strudel parameters at specific beats/cycles.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import json

# Automation event types
EVENT_TYPES = {
    "set": "Instantly set a parameter value",
    "sweep": "Gradually change from current to target over duration",
    "toggle": "Toggle a boolean (on/off) state",
    "trigger": "One-shot trigger (e.g., fill, transition)",
    "random": "Set to random value within range",
}

# Controllable parameters per voice
VOICE_PARAMS = {
    "bass": ["gain", "lpf", "hpf", "attack", "decay", "distort", "room"],
    "mid": ["gain", "lpf", "hpf", "attack", "decay", "phaser", "room", "delay"],
    "high": ["gain", "lpf", "hpf", "attack", "room", "delay", "crush"],
    "drums": ["gain", "room", "crush", "swing"],
    "master": ["gain", "lpf", "hpf", "room"],
}

# Parameter ranges for validation
PARAM_RANGES = {
    "gain": (0.0, 2.0),
    "lpf": (20, 20000),
    "hpf": (20, 2000),
    "attack": (0.001, 0.5),
    "decay": (0.01, 2.0),
    "sustain": (0.0, 1.0),
    "release": (0.01, 2.0),
    "room": (0.0, 1.0),
    "delay": (0.0, 1.0),
    "distort": (0.0, 1.0),
    "phaser": (0.0, 1.0),
    "crush": (1, 16),
    "swing": (0.0, 0.5),
}


@dataclass
class AutomationEvent:
    """A single automation event at a specific beat."""
    beat: float  # Beat number (0-based, can be fractional)
    voice: str  # "bass", "mid", "high", "drums", "master"
    param: str  # Parameter name (gain, lpf, etc.)
    type: str = "set"  # "set", "sweep", "toggle", "trigger", "random"
    value: Optional[float] = None  # Target value for "set"
    from_value: Optional[float] = None  # Start value for "sweep"
    to_value: Optional[float] = None  # End value for "sweep"
    duration: Optional[float] = None  # Duration in beats for "sweep"
    curve: str = "linear"  # "linear", "sine", "exp", "log"
    min_value: Optional[float] = None  # For "random" type
    max_value: Optional[float] = None  # For "random" type

    def validate(self) -> bool:
        """Validate the event parameters."""
        if self.voice not in VOICE_PARAMS and self.voice != "master":
            return False
        if self.param not in PARAM_RANGES:
            return False

        min_val, max_val = PARAM_RANGES[self.param]

        if self.type == "set" and self.value is not None:
            return min_val <= self.value <= max_val
        elif self.type == "sweep":
            if self.to_value is not None:
                return min_val <= self.to_value <= max_val
        elif self.type == "random":
            if self.min_value is not None and self.max_value is not None:
                return min_val <= self.min_value <= max_val and min_val <= self.max_value <= max_val

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Section:
    """A musical section with its own parameter settings."""
    name: str  # "intro", "verse", "chorus", "drop", "breakdown", "outro"
    start_beat: float
    end_beat: float
    energy: float = 0.5  # 0.0-1.0, affects overall intensity
    voices: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start_beat,
            "end": self.end_beat,
            "energy": self.energy,
            "voices": self.voices
        }


@dataclass
class AutomationTimeline:
    """Complete automation timeline for a track."""
    bpm: float
    total_beats: int
    sections: List[Section] = field(default_factory=list)
    events: List[AutomationEvent] = field(default_factory=list)

    # Global settings
    initial_state: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_event(self, event: AutomationEvent) -> bool:
        """Add an event if valid."""
        if event.validate():
            self.events.append(event)
            return True
        return False

    def add_section(self, section: Section):
        """Add a section."""
        self.sections.append(section)

    def get_events_at_beat(self, beat: float, tolerance: float = 0.1) -> List[AutomationEvent]:
        """Get all events at a specific beat."""
        return [e for e in self.events if abs(e.beat - beat) < tolerance]

    def get_section_at_beat(self, beat: float) -> Optional[Section]:
        """Get the section at a specific beat."""
        for section in self.sections:
            if section.start_beat <= beat < section.end_beat:
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bpm": self.bpm,
            "total_beats": self.total_beats,
            "initial_state": self.initial_state,
            "sections": [s.to_dict() for s in self.sections],
            "events": [e.to_dict() for e in self.events]
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutomationTimeline':
        """Create from dictionary."""
        timeline = cls(
            bpm=data.get("bpm", 120),
            total_beats=data.get("total_beats", 64),
            initial_state=data.get("initial_state", {})
        )

        for section_data in data.get("sections", []):
            section = Section(
                name=section_data["name"],
                start_beat=section_data["start"],
                end_beat=section_data["end"],
                energy=section_data.get("energy", 0.5),
                voices=section_data.get("voices", {})
            )
            timeline.add_section(section)

        for event_data in data.get("events", []):
            event = AutomationEvent(
                beat=event_data["beat"],
                voice=event_data["voice"],
                param=event_data["param"],
                type=event_data.get("type", "set"),
                value=event_data.get("value"),
                from_value=event_data.get("from_value"),
                to_value=event_data.get("to_value"),
                duration=event_data.get("duration"),
                curve=event_data.get("curve", "linear"),
                min_value=event_data.get("min_value"),
                max_value=event_data.get("max_value")
            )
            timeline.add_event(event)

        return timeline

    @classmethod
    def from_json(cls, json_str: str) -> 'AutomationTimeline':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def generate_example_timeline(bpm: float = 136, duration_seconds: float = 60) -> AutomationTimeline:
    """Generate an example automation timeline."""
    total_beats = int(duration_seconds * bpm / 60)

    timeline = AutomationTimeline(
        bpm=bpm,
        total_beats=total_beats,
        initial_state={
            "bass": {"gain": 0.2, "lpf": 400, "hpf": 40},
            "mid": {"gain": 0.6, "lpf": 4000, "hpf": 300},
            "high": {"gain": 0.4, "lpf": 12000, "hpf": 800},
            "drums": {"gain": 0.8, "room": 0.15}
        }
    )

    # Add sections
    timeline.add_section(Section("intro", 0, 16, energy=0.3, voices={
        "bass": {"gain": 0.15, "lpf": 300},
        "drums": {"gain": 0.5}
    }))

    timeline.add_section(Section("buildup", 16, 32, energy=0.6, voices={
        "bass": {"gain": 0.3},
        "mid": {"gain": 0.8}
    }))

    timeline.add_section(Section("drop", 32, 64, energy=1.0, voices={
        "bass": {"gain": 0.5, "lpf": 800},
        "mid": {"gain": 1.2},
        "drums": {"gain": 1.0}
    }))

    # Add automation events
    # Filter sweep during buildup
    timeline.add_event(AutomationEvent(
        beat=16, voice="bass", param="lpf", type="sweep",
        from_value=300, to_value=800, duration=16, curve="exp"
    ))

    # Gain swell
    timeline.add_event(AutomationEvent(
        beat=24, voice="mid", param="gain", type="sweep",
        from_value=0.6, to_value=1.2, duration=8, curve="linear"
    ))

    # Drop hit - instant changes
    timeline.add_event(AutomationEvent(
        beat=32, voice="bass", param="gain", type="set", value=0.5
    ))
    timeline.add_event(AutomationEvent(
        beat=32, voice="drums", param="gain", type="set", value=1.0
    ))

    # Add some variation
    timeline.add_event(AutomationEvent(
        beat=48, voice="high", param="lpf", type="random",
        min_value=8000, max_value=15000
    ))

    return timeline


# LLM prompt template for generating automation
AUTOMATION_PROMPT_TEMPLATE = """
## AUTOMATION TIMELINE FORMAT

Generate a JSON automation timeline for this track. The timeline controls how parameters change over time.

### JSON Schema:
```json
{
  "bpm": 136,
  "total_beats": 64,
  "initial_state": {
    "bass": {"gain": 0.2, "lpf": 400},
    "mid": {"gain": 0.6, "lpf": 4000},
    "high": {"gain": 0.4, "lpf": 12000},
    "drums": {"gain": 0.8}
  },
  "sections": [
    {"name": "intro", "start": 0, "end": 16, "energy": 0.3},
    {"name": "buildup", "start": 16, "end": 32, "energy": 0.6},
    {"name": "drop", "start": 32, "end": 64, "energy": 1.0}
  ],
  "events": [
    {"beat": 16, "voice": "bass", "param": "lpf", "type": "sweep", "from_value": 300, "to_value": 800, "duration": 16},
    {"beat": 32, "voice": "drums", "param": "gain", "type": "set", "value": 1.0}
  ]
}
```

### Event Types:
- **set**: Instantly set parameter to value
- **sweep**: Gradually change over duration beats (curves: linear, sine, exp)
- **toggle**: On/off state
- **random**: Random value in min_value..max_value range

### Voices: bass, mid, high, drums, master

### Parameters:
- gain (0-2): Volume
- lpf (20-20000): Low-pass filter cutoff
- hpf (20-2000): High-pass filter cutoff
- room (0-1): Reverb
- delay (0-1): Delay wet
- distort (0-1): Distortion
- attack (0.001-0.5): Envelope attack
- crush (1-16): Bit crush

### Musical Guidelines:
1. **Intro** (energy 0.2-0.4): Low gain, filtered sounds, minimal drums
2. **Buildup** (energy 0.5-0.7): Gradual filter sweeps, increasing gain
3. **Drop** (energy 0.8-1.0): Full gain, open filters, punchy drums
4. **Breakdown** (energy 0.3-0.5): Strip back, filter down
5. **Outro** (energy 0.2-0.3): Fade out, close filters

### Example Automation Patterns:
- Filter sweep: `{"beat": 16, "voice": "bass", "param": "lpf", "type": "sweep", "to_value": 2000, "duration": 8}`
- Drop hit: `{"beat": 32, "voice": "drums", "param": "gain", "type": "set", "value": 1.0}`
- Breakdown: `{"beat": 48, "voice": "mid", "param": "gain", "type": "sweep", "to_value": 0.3, "duration": 4}`
"""


if __name__ == "__main__":
    # Generate and print example
    timeline = generate_example_timeline(bpm=136, duration_seconds=60)
    print(timeline.to_json())
