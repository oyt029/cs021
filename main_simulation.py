import logging
# from turtle import home
logging.basicConfig(level=logging.INFO, filename='simulation.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.ERROR)
import re
import sys
import os
import json
import math
import copy
import faiss
from datetime import datetime
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.llms import OpenAI
from langchain_community.docstore import InMemoryDocstore
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.utils import mock_now
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from agent_simulation.custom_agent import GenerativeAgent
from agent_simulation.prompts import AgentPrompt
from agent_simulation.utils import (extract_agent_action, 
                                    load_agent_info, 
                                    save_plans, 
                                    format_time_with_date,
                                    )

# from home_bot.homebot_agent import HomebotAgent
from home_bot.homebot_init import homebot
from device_simulation.device_factory import CustomDeviceFactory


# print(get_floor_data(os.path.join(os.getcwd(), 'data', 'envData')))

# ======== Set the OpenAI API key ======== #
os.environ['OPENAI_API_KEY'] = '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'
openai_api_key_embedding = os.getenv('OPENAI_API')
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_api_base='https://api.openai.com/v1'
# openai_api_key = os.getenv('OPENAI_API')
# openai_api_base = "https://openrouter.ai/api/v1"
headers={"HTTP-Referer": "https://localhost:3000/"}
if openai_api_key is None:
    print("API_KEY environment variable not found. Please set the API key.")
    sys.exit()
else:
    print("API_KEY set up correctly!")
    print("+"*20)

# ======== Set the OpenAI AI engine ======== #
# LLM = ChatOpenAI(max_tokens=5000, model_name='gpt-4-0314', temperature=0.0)
# BOT_LLM = ChatOpenAI(max_tokens=2000, model_name='gpt-4-0613', temperature=0.0)
LLM = ChatOpenAI(max_tokens=8000, model_name='gpt-4o-mini', temperature=0.0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)
# LLM = ChatOpenAI(max_tokens=8000, model_name='gpt-3.5-turbo-16k', temperature=0.0, openai_api_key=openai_api_key)
# BOT_LLM = ChatOpenAI(max_tokens=2000, model_name='gpt-4-32k-0314', temperature=0.0, openai_api_key=openai_api_key, openai_api_base=openai_api_base, headers=headers)

# ======== Define the relevance score function ======== #
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

# ======== Define the agent memory ======== #
def create_agent_memory(retriever: bool = False) -> GenerativeAgentMemory:
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key_embedding)
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    memory_retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )
    # Define the agent memory
    agent_memory = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=memory_retriever,
        verbose=False,
        reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
    )
    if retriever:
        return memory_retriever
    else:
        return agent_memory

# ======== Define the agents ======== # 
def agent_init(agent_info: dict) -> GenerativeAgent:
    """Initialize the agent."""
    agent = GenerativeAgent(
        name=agent_info['name'],
        age=agent_info['age'],
        traits=', '.join(agent_info['preference']),
        status=agent_info['curr_room'],
        memory_retriever=create_agent_memory(retriever=True),
        llm=LLM,
        memory=create_agent_memory(retriever=False),
        verbose=True,
    )
    return agent

def env_init():
    device_factory = CustomDeviceFactory("device_simulation/devices.json")
    all_devices = device_factory.show_devices()
    for device_name, dev_model in all_devices.items():
        room_name, dev_name = device_name.split(":")
        
        current_dir = os.getcwd()
        env_data_path = os.path.join(current_dir, 'data', 'envData', f'{room_name}_data.json')
        # open env data
        with open(env_data_path, 'r', encoding='utf-8') as f:
            env_data = json.load(f)

        current_status = device_factory.check_device_status(device_name)
        if "power" in current_status:
            if current_status["power"] == "off":
                env_data[dev_name] = {"power": "off"}
            else:
                env_data[dev_name] = current_status
        else:
            env_data[dev_name] = current_status

        # save env data
        with open(env_data_path, 'w', encoding='utf-8') as f:
            json.dump(env_data, f, indent=4)
    return device_factory

def init_device_steps(device_steps):
    device_steps["init"] = []
    current_dir = os.getcwd()
    env_data_folder_path = os.path.join(current_dir, 'data', 'envData')
    all_env_data_files = sorted(os.listdir(env_data_folder_path))
    for file_name in all_env_data_files:
        if file_name.endswith(".json"):
            env_data_path = os.path.join(current_dir, 'data', 'envData', file_name)
            with open(env_data_path, 'r', encoding='utf-8') as f:
                env_data = json.load(f)
            device_steps["init"].append(env_data)
    return device_steps
    
def update_device_steps(device_steps: dict, steps: list, agent_type: str, plan: str, act: str):
    for step in steps:
        if step[0].tool == 'Device Control in Multiple Rooms' or step[0].tool == 'Device Control':
            if isinstance(step[1], list) or isinstance(step[1], dict):
                device_steps[plan][agent_type].append(step[1])

    return device_steps


############################################################################################################
#                                                                                                          #
#                                                                                                          #
#   Agent simulation starts below                                                                          #
#                                                                                                          #
#                                                                                                          #
############################################################################################################

def run():
    # ======== Init Env Data ======== #
    device_factory = env_init()
    support_models = list(np.unique([x.split(":")[1] for x in device_factory.show_devices().keys()]))

    # ======== Init Homebot ======== #

    agent_info_dir = os.path.join(os.getcwd(), 'data', 'agentData')
    # ======== Init Emma ======== #
    emma_info_dir = os.path.join(agent_info_dir, 'emma.json')
    emma_info = load_agent_info(emma_info_dir)
    emma = agent_init(emma_info)
    emma_life_style = ", ".join(emma_info['life_style'])
    # set emma morning prompt
    emma_morning_prompt = AgentPrompt.EMMA_MORNINGS_PROMPT
    # set emma evening prompt
    emma_evening_prompt = AgentPrompt.EMMA_EVENINGS_PROMPT.format(emma_life_style)

    # print(emma_evening_prompt)
    # We can add memories directly to the memory object
    emma_observations = [
        "Tommy is your son.",
        "Jason is your husband.",
        "Emma is hungry.",
        "Emma needs to get ready for work.",
    ]

    for observation in emma_observations:
        with mock_now(datetime(2023, 9, 1, 6, 0)):
            emma.memory.add_memory(observation)

    # print(emma.change_env(preference=" ".join(emma_info['preference']), curr_room=emma_info['curr_room']))

    emma_morning_plan = emma.make_plan(emma_morning_prompt)
    emma_evening_plan = emma.make_plan(emma_evening_prompt)
    plan_list, emma_plan, emma_info = save_plans(emma_morning_plan + emma_evening_plan, emma_info_dir)
    # plan_list, emma_plan = save_plans(emma_morning_plan, 'agentData/emma.json')

    # print(emma_morning_plan)
    # print("="*15)
    # print(emma_evening_plan)

    print("====", "emma_plan", "====")
    print(*plan_list, sep="\n")
    print("+"*15)

    # ======== Init Jason ======== #
    jason_info_dir = os.path.join(agent_info_dir, 'jason.json')
    jason_info = load_agent_info(jason_info_dir)
    jason = agent_init(jason_info)
    jason_life_style = ", ".join(jason_info['life_style'])
    emma_morning_plan = " ".join([emma_info['first_name'], emma_morning_plan])
    emma_evening_plan = " ".join([emma_info['first_name'], emma_evening_plan])

    # set jason morning prompt
    jason_morning_prompt = AgentPrompt.JASON_MORNINGS_PROMPT.format(emma_morning_plan)
    # set jason evening prompt
    jason_evening_prompt = AgentPrompt.JASON_EVENINGS_PROMPT.format(emma_evening_plan, jason_life_style)

    # We can add memories directly to the memory object
    jason_observations = [
        "Emma is your wife.",
        "Tommie is your son.",
        "Jason is hungry.",
        "Jason needs to get ready for work.",
    ]

    for observation in jason_observations:
        with mock_now(datetime(2023, 9, 1, 6, 30)):
            jason.memory.add_memory(observation)

    # print(jason.change_env(preference=" ".join(jason_info['preference']), curr_room=jason_info['curr_room']))

    jason_morning_plan = jason.make_plan(jason_morning_prompt)
    jason_evening_plan = jason.make_plan(jason_evening_prompt)
    plan_list, jason_plan, jason_info = save_plans(jason_morning_plan + jason_evening_plan, jason_info_dir)
    # plan_list, jason_plan = save_plans(jason_morning_plan, 'agentData/jason.json')


    print("====", "jason_plan", "====")
    print(*plan_list, sep="\n")
    print("+"*15)

    # ======== Init Tommie ======== #
    tommie_info_dir = os.path.join(agent_info_dir, 'tommie.json')
    tommie_info = load_agent_info(tommie_info_dir)
    tommie = agent_init(tommie_info)
    tommie_life_style = ", ".join(tommie_info['life_style'])
    jason_morning_plan = " ".join([jason_info['first_name'], jason_morning_plan])
    jason_evening_plan = " ".join([jason_info['first_name'], jason_evening_plan])
    emma_morning_plan = " ".join([emma_info['first_name'], emma_morning_plan])
    emma_evening_plan = " ".join([emma_info['first_name'], emma_evening_plan])

    # set tommie morning prompt
    tommie_morning_prompt = AgentPrompt.TOMMIE_MORNINGS_PROMPT.format(emma_morning_plan+jason_morning_plan)
    # set tommie evening prompt
    tommie_evening_prompt = AgentPrompt.TOMMIE_EVENINGS_PROMPT.format(emma_evening_plan+jason_evening_plan, tommie_life_style)

    # We can add memories directly to the memory object
    tommie_observations = [
        "Emma is your mum.",
        "Jason is your dad.",
        "Tommie is hungry.",
        "Tommie needs to get ready for school.",
    ]

    for observation in tommie_observations:
        with mock_now(datetime(2023, 9, 1, 7, 0)):
            tommie.memory.add_memory(observation)

    # print(tommie.change_env(preference=" ".join(tommie_info['preference']), curr_room=tommie_info['curr_room']))

    tommie_morning_plan = tommie.make_plan(tommie_morning_prompt)
    tommie_evening_plan = tommie.make_plan(tommie_evening_prompt)
    plan_list, tommie_plan, tommie_info = save_plans(tommie_morning_plan + tommie_evening_plan, tommie_info_dir)
    # plan_list, tommie_plan = save_plans(tommie_morning_plan, 'agentData/tommie.json')

    print("====", "tommie_plan", "====")
    print(*plan_list, sep="\n")
    print("+"*15)

    # ======== Start all agents plan in a single loop ======== #
    persons = {
        "Emma": [emma, emma_info],
        "Jason": [jason, jason_info],
        "Tommie": [tommie, tommie_info]
    }

    combined_plans = emma_plan['plan'] + jason_plan['plan'] + tommie_plan['plan']

    # Create a deep copy of combined_plans
    copied_combined_plans = copy.deepcopy(combined_plans)

    # Add start_time_obj and end_time_obj to each activity in the copied list
    for act in copied_combined_plans:
        act["start_time_obj"] = datetime.strptime(act["start_time"], "%H:%M").time()
        act["end_time_obj"] = datetime.strptime(act["end_time"], "%H:%M").time()

    # Sort the copied list
    sorted_acts = sorted(copied_combined_plans, key=lambda x: x["start_time_obj"])

    # Loop through sorted acts and print the output
    agent_steps = {}
    homebot_steps = {}
    device_steps = {}
    index = 0
    all_preference = f"Emma: {' '.join(emma_info['preference'])}\nTommie: {' '.join(tommie_info['preference'])}\nJason: {' '.join(jason_info['preference'])}"

    print(all_preference)

    agent_steps_path = os.path.join(os.getcwd(), 'data', 'simulationData', 'agent_steps.json')
    homebot_steps_path = os.path.join(os.getcwd(), 'data', 'simulationData', 'homebot_steps.json')
    device_steps_path = os.path.join(os.getcwd(), 'data', 'simulationData', 'device_steps.json')

    device_steps = init_device_steps(device_steps)

    for act in sorted_acts:
        start_time = time.time()  # 新增
        #TODO: update the code for 3 persons
        person_name = act["name"]
        idle_person_names = [name for name in persons if name != person_name]

        print("-"*15)
        plan = f"{person_name}: [{act['start_time']} - {act['end_time']}] {act['action']}- {act['act_place']}"
        # print(plan)

        tmp_start_hour, tmp_start_min = act['start_time'].split(":")
        tmp_start_hour = int(tmp_start_hour)
        tmp_start_min = int(tmp_start_min)

        with mock_now(datetime(2023, 9, 1, tmp_start_hour, tmp_start_min)):
            if person_name == "Emma":
                emma.memory.add_memory(plan)
            elif person_name == "Jason":
                jason.memory.add_memory(plan)
            elif person_name == "Tommie":
                tommie.memory.add_memory(plan)
        
        person_object = persons.get(person_name)[0]
        person_info = persons.get(person_name)[1]
        # print(person_info)
        idle_person_info_1 = persons.get(idle_person_names[0])[1]
        idle_person_info_2 = persons.get(idle_person_names[1])[1]

        if person_object:
            # Write current room to the agent info
            person_info['curr_room'] = act['act_place']  

            # Write current time to the agent info
            if person_info['curr_time']:
                curr_date_time = person_info['curr_time']
                new_date_time = format_time_with_date(plan, curr_date_time)
                print("====", new_date_time, "====")
                person_info['curr_time'] = new_date_time

            # Write current task to the agent info
            person_info['curr_task'] = act['action']

            with open(os.path.join(agent_info_dir, f"{person_name.lower()}.json"), 'w') as f:
                json.dump(person_info, f, indent=4)

            all_status = (f"{person_name} is in: {act['act_place']}, the current status is: {act['action']}",
                        f"{idle_person_names[0]} is in: {idle_person_info_1['curr_room']}, the current status is: {idle_person_info_1['curr_task']}",
                        f"{idle_person_names[1]} is in: {idle_person_info_2['curr_room']}, the current status is: {idle_person_info_2['curr_task']}")

            print(all_status)

            device_steps[plan] = {
                "start_time": act["start_time"],
                "end_time": act["end_time"],
                "bot": [],
                person_name: []
            }
            # TODO: let the homebot change the environment and provide the feedback first
            bot_response, bot_steps = homebot.change_env(all_preference=all_preference, 
                                                        curr_time=person_info['curr_time'],
                                                        family_status=all_status,
                                                        smart_devices=support_models)

            device_steps = update_device_steps(device_steps, bot_steps, 'bot', plan, act)

            # let the agent change the environment and provide the feedback
            response, steps = person_object.change_env(preference=" ".join(person_info['preference']), 
                                                    curr_person=person_name,
                                                    curr_room=act['act_place'],
                                                    curr_time=person_info['curr_time'],
                                                    curr_task=act['action'],
                                                    smart_devices=support_models)

            device_steps = update_device_steps(device_steps, steps, person_name, plan, act)

            steps = extract_agent_action(steps)
            bot_steps = extract_agent_action(bot_steps)

            agent_steps[plan] = [response, steps]
            homebot_steps[plan] = [bot_response, bot_steps]

        # if index == 5:
        #     break
        end_time = time.time()  # 新增
        duration = end_time - start_time  # 新增
        logging.info(
            f"{person_name} executed action '{act['action']}' in '{act['act_place']}' taking {duration:.2f} seconds.")  # 新增
        index += 1

    # print(homebot_steps)

    with open(agent_steps_path, 'w') as f:
        json.dump(agent_steps, f, indent=4)

    with open(homebot_steps_path, 'w') as f:
        json.dump(homebot_steps, f, indent=4)

    with open(device_steps_path, 'w') as f:
        json.dump(device_steps, f, indent=4)

    print("Agent_steps is saved into agent_steps.json")
    print("Homebot_steps is saved into homebot_steps.json")
    print("Device_steps is saved into device_steps.json")


if __name__ == "__main__":
    run()