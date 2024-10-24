import json
import os
from typing import Any, Dict, List, Optional, Union
import openai
import dotenv
import pathlib

from dice import roll_dice

dotenv.load_dotenv()

thisdir = pathlib.Path(__file__).parent

client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

def create_ttrpg(inspiration: str) -> str:
    example = (thisdir / "ttrpg-example.md").read_text()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a one-page TTRPG creator. You have to create a one-page TTRPG given an inspiration (idea). "
            )
        },
        {
            "role": "user",
            "content": "Fantasy, magical realms, etc."
        },
        {
            "role": "assistant",
            "content": example
        },
        {
            "role": "user",
            "content": inspiration
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

Notes = Dict[str, Union["Notes", str, float, int]]

def update_notes(notes: Notes, key: str, value: Union[Notes, str, float, int]) -> Notes:
    key_parts = key.split(".")
    if len(key_parts) == 1:
        notes[key] = value
    else:
        update_notes(notes[key_parts[0]], ".".join(key_parts[1:]), value)

def get_notes(notes: Notes, key: Optional[str] = None) -> Union[Notes, str, float, int]:
    if key is None:
        return notes
    key_parts = key.split(".")
    if len(key_parts) == 1:
        return notes[key]
    else:
        return get_notes(notes[key_parts[0]], ".".join(key_parts[1:]))

def play_ttrpg(ttrpg_path: pathlib.Path,
               notes_path: pathlib.Path,
               messages_path: pathlib.Path) -> str:
    if not ttrpg_path.exists():
        ttrpg = create_ttrpg("Anything!")
        ttrpg_path.write_text(ttrpg)
        if notes_path.exists():
            os.unlink(notes_path)
        if messages_path.exists():
            os.unlink(messages_path)

    notes = {}
    if notes_path.exists():
        notes = json.loads(notes_path.read_text())

    ttrpg = ttrpg_path.read_text()

    if not messages_path.exists():
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the GM of the following TTRPG: \n"
                    f"{ttrpg}\n\n"
                    "Your job is to play the TTRPG and keep track of notes. "
                    "Start by setting up the game with the user and then play the game. "
                    "Be descriptive and fun, don't tell them what you're doing behind the scenes (e.g., challenge ratings, etc.) "
                    "You should roll for the user, but let them make decisions (e.g., if they say they want to attack, you say 'okay I'm going to roll a d6 to see if you hit'). "
                    "Always tell them what you rolled, why you're rolling, and what the result is. "
                    "Keep track of things like abilities, user choices, stats, hit points, etc. in your notes. "
                    "Don't give the user lists of options on what to do next. "
                    "Instead ask them what they want to do and let them respond in natural language. "
                )
            },
            {
                "role": "user",
                "content": "Let's play!"
            }
        ]
        messages_path.write_text(json.dumps(messages, indent=4, ensure_ascii=False))
    else:
        messages = json.loads(messages_path.read_text())
    
    tools = {
        "roll_dice": {
            "type": "function",
            "function": {
                "name": "roll_dice",
                "description": "Roll dice.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dice_str": {
                            "type": "string",
                            "description": (
                                "String representing the dice roll. For example, '4d6' means roll 4 six-sided dice. "
                                "Must be in the format '<number_of_dice>d<number_of_sides>'. "
                                "For example, '2d10' means roll 2 ten-sided dice."
                            )
                        },
                        "keep": {
                            "type": "string",
                            "description": (
                                "String representing how many dice to keep. "
                                "For example, 'highest_3' means keep the highest 3 dice and 'lowest_2' means keep the lowest 2 dice. "
                                "Must be 'highest_<n>' or 'lowest_<n>' where <n> is a positive integer less than the number of dice rolled."
                            )
                        }
                    },
                    "required": ["dice_str"],
                    "additionalProperties": False,
                },
            }
        },
        "update_notes": {
            "type": "function",
            "function": {
                "name": "update_notes",
                "description": "Update notes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "object",
                            "description": "Notes object."
                        },
                        "key": {
                            "type": "string",
                            "description": "Key to update."
                        },
                        "value": {
                            "description": "Value to set.",
                        }
                    },
                    "required": ["notes", "key", "value"],
                    "additionalProperties": False,
                },
            }
        },
        "get_notes": {
            "type": "function",
            "function": {
                "name": "get_notes",
                "description": "Get notes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "object",
                            "description": "Notes object."
                        },
                        "key": {
                            "type": "string",
                            "description": "Key to get."
                        }
                    },
                    "required": ["notes", "key"],
                    "additionalProperties": False,
                },
            }
        },
        "get_user_response": {
            "type": "function",
            "function": {
                "name": "prompt_user",
                "description": (
                    "Use this to indicate that the user should be prompted for a response. "
                    "The user cannot respond until this tool is called. "
                    "In other words, this is used by the GM to mark the end of their turn. "
                )
            }
        }
    }

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[
                tools['roll_dice'],
                tools['update_notes'],
                tools['get_notes'],
                tools['get_user_response']
            ]
        )

        message = response.choices[0].message
        messages.append(json.loads(message.model_dump_json()))

        (thisdir / "messages.json").write_text(json.dumps(messages, indent=4, ensure_ascii=False))

        do_prompt_user = len(message.content or "") != 0
        if message.tool_calls:
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "roll_dice":
                    try:
                        result = roll_dice(args["dice_str"], args.get("keep"))
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"success": True, "result": result}, ensure_ascii=False),
                            "tool_call_id": tool_call.id
                        })
                        do_prompt_user = False # don't prompt user after rolling dice
                    except ValueError as e:
                        result = str(e)
                        messages.append({
                            "role": "tool",
                            "content": json.dumps({"success": False, "error": result}, ensure_ascii=False),
                            "tool_call_id": tool_call.id
                        })
                elif tool_call.function.name == "update_notes":
                    update_notes(notes, args["key"], args["value"])
                    notes_path.write_text(json.dumps(notes, indent=4, ensure_ascii=False))
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"success": True}, ensure_ascii=False),
                        "tool_call_id": tool_call.id
                    })
                elif tool_call.function.name == "get_notes":
                    result = get_notes(notes, args["key"])
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"success": True, "result": result}, ensure_ascii=False),
                        "tool_call_id": tool_call.id
                    })

        if message.content:
            print(f"\nGM: {message.content}\n")

        if do_prompt_user:
            user_response = input("You: ")
            messages.append({
                "role": "user",
                "content": user_response
            })
        # save notes and messages
        notes_path.write_text(json.dumps(notes, indent=4, ensure_ascii=False))
        (thisdir / "messages.json").write_text(json.dumps(messages, indent=4, ensure_ascii=False))



def main():
    play_ttrpg(
        ttrpg_path=thisdir / "ttrpg.md",
        notes_path=thisdir / "notes.json",
        messages_path=thisdir / "messages.json"
    )

if __name__ == "__main__":
    main()
