import sys
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import operator, os
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
import re
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.runnables.config import RunnableConfig

class POMPackageNames(BaseModel):
    """"""
    package_names: list[str] = Field(..., description="List of Java packages to include in the POM file")

class code(BaseModel):
    """Code output"""
    code: str = Field(description="Full Java code including import statemens.")

class xml(BaseModel):
    """XML output"""
    code: str = Field(description="Full XML for configuration file.")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    def __init__(self):
        self.recursion_count = 0

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        """graph.add_conditional_edges(
                from_node="llm",
                to_node="action",
                condition=lambda state: state.recursion_count >= graph.config.get('recursion_limit', 5)
        )"""
        #graph.config.get('recursion_limit', 100)
        graph.set_entry_point("llm")

        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        # print("Calling open AI")
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        # print("in exists action")
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            # print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        # print("Back to the model!")
        return {'messages': results}


class CodeFile(BaseModel):
    file_name: str = Field(..., description="Name of the file")
    content: str = Field(..., description="Content of the file")
@tool(args_schema=CodeFile)
def save_file(file_name: str, content: str):
    """
    Save the generated code to a file
    """
    with open(file_name, "w") as f:
        f.write(content)
    return "File saved successfully"


class Directory(BaseModel):
    directory_path: str = Field(..., description="Path of the directory to create")
@tool(args_schema=Directory)
def create_directory(directory_path: str):
    """
    Create a directory at the given path
    """
    os.makedirs(directory_path, exist_ok=True)
    return "Directory created successfully"

import subprocess
class MavenRun(BaseModel):
    command: str = Field(..., description="Maven command to run. Example 'clean install', 'compile', 'test")
    project_dir: str = Field(..., description="Path to the directory containing the Maven project")

@tool(args_schema=MavenRun)
def run_maven_command(command, project_dir):
    """
    Executes a Maven command in the specified project directory.

    :param command: Maven command to run (e.g., 'clean install').
    :param project_dir: Path to the directory containing the Maven project.
    """
    try:
        # Construct the full Maven command
        full_command = f"/opt/homebrew/Cellar/maven/3.9.7/bin/mvn {command}"

        # Run the command and capture the output
        result = subprocess.run(
            full_command,
            cwd=project_dir,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Print the output and error (if any)
        print("Output:\n", result.stdout)
        if result.stderr:
            print("Errors:\n", result.stderr)

        print(f"Maven command '{command}' executed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the Maven command: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Errors: {e.stderr}")
        return e.returncode, e.output, e.stderr


class CreateProject(BaseModel):
    base_dir: str = Field(..., description="Base directory for the project")
    project_name: str = Field(..., description="Name of the project")
    pom_content: str = Field(..., description="Content of the pom.xml file")

@tool(args_schema=CreateProject)
def create_maven_project(base_dir, project_name, pom_content):
    """
    Create a Maven project with the specified name and pom.xml content in the given base directory.
    """
    # Create the base directory for the project
    print("Creating maven project")
    project_dir = os.path.join(base_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Define standard Maven directories
    directories = [
        "src/main/java/com/github/dheerajhegde/esql",
        "src/main/resources",
        "src/test/java/com/github/dheerajhegde/esql",
        "src/test/resources"
    ]

    # Create the directories
    for directory in directories:
        os.makedirs(os.path.join(project_dir, directory), exist_ok=True)

    # Write the pom.xml file
    pom_file_path = os.path.join(project_dir, "pom.xml")
    with open(pom_file_path, 'w') as pom_file:
        pom_file.write(pom_content)

    print(f"Project {project_name} created successfully at {project_dir}")
    return project_dir


class Java(BaseModel):
    functionality: str = Field(..., description="Functionality to be replicated in Java")
    guidance: str = Field(..., description="Specific guidance to use while creating the Java code")


@tool(args_schema=Java)
def create_spring_code(functionality: str, guidance: str):
    """
    Create spring code to meet functional requirements
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    model_name = "codestral-latest"
    model = ChatMistralAI(api_key=api_key, model=model_name)

    prompt = f"""
        Create java code for the functionality provided: {functionality}
        - Make sure to comment the code as per Java coding standards
        - Ensure the code is correctly formatted and follows the standard Java Spring conventions

        Use below guidance as you create the code
        {guidance}

    Your response should only be the Java code for the entity definition. Nothing else. 
    """
    code_gen_chain = model.with_structured_output(code)
    response = code_gen_chain.invoke("create java spring code")
    print(response)
    return response



model = ChatOpenAI(model='gpt-4o', openai_api_key=os.getenv("OPENAI_API_KEY"))
with open("./ESQL/WeatherServiceModule.esql", "r") as f:
    esql_content = f.read()
with open("./ESQL/WeatherService.msgflow", "r") as f:
    msgflow_content = f.read()

package_list_response = ["javax.jms", "org.springframework.boot", "org.springframework.boot:spring-boot-starter-web", "org.springframework.boot:spring-boot-starter-test"]
print(package_list_response)

maven_project_prompt = """
    Create one maven project with Java 19. Project name is {project_name} and directory under which to create project is {project_root}
        - Create a maven pom.xml file with dependencies listed here -->  {package_list_response}
        - tool provided is 'create_maven_project'
        - Add standard dependencies like spring-boot-starter, spring-boot-starter-web, spring-boot-starter-test, etc.
"""

project_root = "/Users/dheerajhegde/Documents/Code/0 Working POCs and Applications/"
project_name = "ESQLJavaProject"
tools = [create_maven_project]
maven_prompt_text = maven_project_prompt.format(project_root=project_root, project_name=project_name, package_list_response=package_list_response)
abot = Agent(model, tools, system=maven_prompt_text)
thread = {"configurable": {"thread_id": "1"}}
project_dir = abot.graph.invoke(
    input={"messages":
               [HumanMessage(content=[{"type": "text", "text": "Create Java Spring Boot application skeleton. "}])],
           "thread": thread
           }
)
print(project_dir)

prompt = """
        
        "Please analyze the provided WeatherService.msgflow and WeatherServiceModule.esql files and translate their functionalities into detailed pseudocode. Follow the steps below for each file type to ensure a comprehensive translation:
        
        1. Message Flow Analysis and Pseudocode Translation:
        
        Understand the Message Flow Structure:
        Identify all nodes in the message flow, such as Input Nodes, Compute Nodes, Filter Nodes, and Output Nodes.
        Describe the connections and data flow between these nodes.
        Functional Description of Each Node:
        Explain the role of each node in the data processing sequence.
        Document any specific transformations, routing, or business logic applied at each node.
        Generate Pseudocode:
        Convert the message flow into high-level pseudocode.
        Include a step-by-step representation of data processing from input to output.
        Incorporate error handling and logging mechanisms present in the flow.
        
        2. ESQL Code (WeatherServiceModule.esql) Analysis and Pseudocode Translation:
        
        Understand ESQL Code Logic:
        Break down the ESQL code to understand how it processes and manipulates messages.
        Identify any database interactions, external system calls, and business rules applied.
        Functional Description of ESQL Statements:
        Describe the purpose and functionality of key ESQL statements and functions.
        Explain how data is being transformed and validated within the code.
        Generate Pseudocode:
        Translate the ESQL code into detailed pseudocode.
        Provide a step-by-step logical representation of data processing and transformations.
        Ensure that the pseudocode includes conditional logic, loops, and data manipulation steps as per the ESQL code.
        Guidelines for Pseudocode:
        
        Use clear and simple language to describe each step.
        Ensure that the pseudocode is high-level but detailed enough to capture the essence of the original logic.
        Avoid using specific syntax from any programming language; focus on the logic and flow.
        
        msgflow_content: {msgflow_content}
        sql_content: {esql_content}
        
        Save the psuedocode in file WeatherServicePseudocode.txt in current working directory. Return the path of the file.
    """




tools = [save_file]
prompt_text = prompt.format(
    project_dir=project_dir,
    esql_content=esql_content,
    msgflow_content=msgflow_content)

config = RunnableConfig(
            recursion_limit=50,  # Set the recursion limit
        )
abot = Agent(model, tools, system=prompt_text)
thread = {"configurable": {"thread_id": "1"}}
response = abot.graph.invoke(
    config=config,
    input={
        "messages": [HumanMessage(content=[{"type": "text", "text": "Create psuedocode and save it to a file."}])],
        "thread": thread
    }
)
print(response["messages"][-1].content)
psuedocode_file = response["messages"][-1].content

with open("./WeatherServicePseudocode.txt", "r") as f:
    psuedocode = f.read()

prompt = """
    Create Java code for the functionality provided: {psuedocode}
    
    Using the provided pseudocode, translate the logic and functionality into Java and Spring Boot code. Follow the steps below to ensure a comprehensive and accurate translation:

    1. Understand the Pseudocode Structure:
    
    Review the pseudocode to grasp the overall flow and functionality.
    Identify key components and operations, such as data processing, routing, transformations, and external interactions.
    
    2. Map Pseudocode Components to Java/Spring Boot Elements:
    
    Determine which parts of the pseudocode correspond to specific Java classes, methods, or Spring Boot components.
    Identify where Controllers, Services, Repositories, and other Spring Boot components should be implemented.
    
    3. Translate Pseudocode into Java and Spring Boot:
    
    Controllers: Map inputs and outputs to RESTful API endpoints using Spring Boot controllers.
    Services: Implement business logic and data processing steps within service classes.
    Repositories: Define database interactions using Spring Data JPA or appropriate data access mechanisms.
    Data Transfer Objects (DTOs): Create DTOs or models for data structures used in the flow.
    Error Handling: Implement error handling mechanisms and logging as per the pseudocode.
    Configuration: Set up necessary configuration properties and beans in Spring Boot.
    
    4. Code Translation Guidelines:
    
    Use clear and appropriate naming conventions for classes, methods, and variables.
    Ensure code is modular and adheres to best practices for Spring Boot applications.
    Include necessary imports and annotations to integrate with the Spring Boot framework.
    Add comments to explain key parts of the code and any complex logic.
    
    Save Java code into package com.github.dheerajhegde.esql in the {project_dir} directory. There will be many files to save. 
    
    Desired Output:
            Java Code:
            - Java Spring Code for the appllication
            Configuration:
            - Use application.properties or application.yml for configuration details.
            Testing:
            - Provide basic unit tests for controllers and services using JUnit.
            - Include integration tests to verify the full HTTP request/response cycle.
            
        After you have write the needed Java code, run the Maven command 'clean install' in the project directory to build the project.
        Go back to code generation or update the pom.xml incase any errors are encountered.
    """

tools = [create_spring_code, run_maven_command, save_file, create_directory]
prompt_text = prompt.format(
    project_dir=project_dir,
    psuedocode=psuedocode)

config = RunnableConfig(
    recursion_limit=50,  # Set the recursion limit
)
abot = Agent(model, tools, system=prompt_text)
thread = {"configurable": {"thread_id": "1"}}
response = abot.graph.invoke(
    config=config,
    input={
        "messages": [HumanMessage(content=[{"type": "text", "text": "Create psuedocode and save it to a file."}])],
        "thread": thread
    }
)
print(response["messages"][-1].content)