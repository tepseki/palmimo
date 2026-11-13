"""Compliment app: when a face appears, compliment the person's style aloud."""

import argparse
import base64
import signal
import time
from types import FrameType

import cv2
import numpy as np
from openai import OpenAI, OpenAIError

from palmimo_sdk import (
    FaceDisplay,
    HeadCamera,
    OpenAiEngine,
    Palmimo,
    Speaker,
    SpeakerConfig,
)

DETECT_WIDTH = 480
LOOP_INTERVAL_S = 0.1

SYSTEM_PROMPT = """あなたは、卓上ロボットとして目の前の人に話しかけます。
画像に写っている人の「服装・髪型・小物・表情・雰囲気」から、
1つだけ具体的に良いところを見つけ、日本語で30文字以内の短い褒め言葉にしてください。
次のことは絶対に言及しないでください:
年齢、性別、体型、肌、人種・国籍、健康状態、容姿の優劣や他人との比較。
人物がはっきり写っていない場合は「SKIP」とだけ返してください。
褒め言葉の本文だけを出力し、前置きや説明は書かないでください。"""


def count_faces(cascade: cv2.CascadeClassifier, frame: np.ndarray) -> int:
    """Count frontal faces in a frame, detecting on a downscaled copy.

    Args:
        cascade: The Haar cascade classifier for frontal faces.
        frame: A BGR camera frame.

    Returns:
        The number of faces found.
    """
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (DETECT_WIDTH, int(height * DETECT_WIDTH / width)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return len(faces)


def make_compliment(client: OpenAI, model: str, frame: np.ndarray) -> str:
    """Ask the vision model for a short compliment about the person in the frame.

    Args:
        client: The OpenAI client.
        model: The vision-capable model name.
        frame: A BGR camera frame.

    Returns:
        The compliment text, or "SKIP" if no person is clearly visible.
    """
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode the frame as JPEG")
    image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この人を褒めてください。"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def speak(robot: Palmimo, text: str) -> None:
    """Speak text and wait until it has finished, reporting a TTS failure.

    Args:
        robot: The connected robot.
        text: What to say.
    """
    handle = robot.say(text, lang="ja")
    if handle is None:
        return
    handle.join()
    if handle.error is not None:
        print(f"Speech failed: {handle.error}")


def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
    """Turn SIGTERM (the Portal's stop) into KeyboardInterrupt so cleanup runs."""
    raise KeyboardInterrupt


def main() -> None:
    """Watch the camera and compliment people who stay in view."""
    parser = argparse.ArgumentParser(description="Face-triggered compliment app")
    parser.add_argument("--model", default="gpt-4o-mini", help="Vision model name")
    parser.add_argument("--cooldown", type=float, default=30.0, help="Seconds between compliments")
    parser.add_argument("--stable-frames", type=int, default=5, help="Frames a face must persist")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    client = OpenAI()  # Reads OPENAI_API_KEY from the environment
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # No driver is attached, so no servo moves: face, voice and camera only.
    robot = Palmimo(
        display=FaceDisplay(),
                speaker=Speaker(SpeakerConfig(lang="ja"), engine=OpenAiEngine(voice="coral")),
        camera=HeadCamera(),
    )

    stable = 0
    last_spoken = float("-inf")
    try:
        with robot:
            assert robot.camera is not None
            speak(robot, "カメラで顔を見て、服装や雰囲気を褒めますね。")
            while True:
                time.sleep(LOOP_INTERVAL_S)
                ok, frame = robot.camera.read()
                if not ok or frame is None:
                    continue
                stable = stable + 1 if count_faces(cascade, frame) > 0 else 0

                cooled = time.monotonic() - last_spoken >= args.cooldown
                if stable < args.stable_frames or not cooled:
                    continue

                last_spoken = time.monotonic()  # Also throttles retries after an error
                stable = 0
                try:
                    text = make_compliment(client, args.model, frame)
                except OpenAIError as exc:
                    print(f"API error: {exc}")
                    continue
                if text and text != "SKIP":
                    print(f"Palmimo: {text}")
                    robot.set_expression("happy", hold_ms=3000)
                    speak(robot, text)
                    last_spoken = time.monotonic()
    except KeyboardInterrupt:
        print("Stopping")


if __name__ == "__main__":
    main()