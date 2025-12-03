

def merge_robot_and_env(robot_xml_string: str, env_xml_path: str) -> str:
    """
    Merges robot and environment XML strings.
    """
    with open(env_xml_path, 'r') as f:
        env_xml = f.read()

        start_tags = ["<worldbody>", "<actuator>", "<default>"]
        end_tags = ["</worldbody>", "</actuator>", "</default>"]

        for i in range(len(start_tags)):
            start_tag = start_tags[i]
            end_tag = end_tags[i]
            if start_tag in robot_xml_string:
                robot_body_content = robot_xml_string.split(start_tag)[1].split(end_tag)[0]
                if start_tag in env_xml:
                    env_xml = env_xml.replace(end_tag, f"{robot_body_content}\n  {end_tag}")
                else:
                    env_xml = env_xml.replace("</mujoco>", f"  {start_tag}\n{robot_body_content}\n  {end_tag}\n</mujoco>")
    return env_xml