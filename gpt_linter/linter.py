import os
import re
import subprocess
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Any, Optional
from subprocess import CompletedProcess

import pandas
from gpt_linter.logger import Logger 

logger = Logger()

class Linter(metaclass=ABCMeta):
    @abstractmethod
    def run_checker(self, file: str, additional_args: List[str], program_path: str, proj_path: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def parse_line(line: str) -> Dict[str, Any]:
        pass

    def get_issues(self, args: Any, override_file: Optional[str] = None) -> List[Dict[str, Any]]:
        out = self.run_checker(args.file if override_file is None else override_file, args.mypy_args, args.mypy_path,
                               args.proj_path)
        logger.info(out)
        issues = [self.parse_line(z) for z in out.split('\n')]
        issues = list(filter(lambda x: x, issues))
        if len(issues) == 0:
            return []
        # Here we unite the issues from different lines
        dfo = pandas.DataFrame(issues)
        changed_message_df = dfo.groupby('Line Number')['Message'].agg(lambda x: '\n'.join(x))
        df_first_row = dfo.groupby('Line Number').first()
        df_first_row['Message'].update(changed_message_df)
        df_first_row = df_first_row.reset_index()
        issues = [dict(r[1]) for r in df_first_row.iterrows()]

        if args.max_errors:
            issues = issues[:args.max_errors]
        if args.error_categories:
            issues = [z for z in issues if z['Category'] in args.error_categories.split(',')]
        return issues


class MyPyLinter(Linter):

    def run_checker(self, file: str, additional_args: List[str], program_path: str, proj_path: str) -> str:
        # Construct the mypy command
        command = [program_path] + additional_args + [file]
        logger.debug("Running mypy command: %s", command)
        # Run mypy command and capture the output
        result: CompletedProcess[str] = subprocess.run(command, capture_output=True, text=True, cwd=os.path.abspath(proj_path))
        # Print the output

        # Decode the output from bytes to string
        output = result.stdout

        # 7-bit C1 ANSI sequences
        ansi_escape = re.compile(r'''
                \x1B  # ESC
                (?:   # 7-bit C1 Fe (except CSI)
                    [@-Z\\-_]
                |     # or [ for CSI, followed by a control sequence
                    \[
                    [0-?]*  # Parameter bytes
                    [ -/]*  # Intermediate bytes
                    [@-~]   # Final byte
                )
            ''', re.VERBOSE)
        clean_output: str = ansi_escape.sub('', output)
        # Print the output
        return clean_output

    @staticmethod
    def parse_line(line: str) -> Dict[str, Any]:
        import re
        pattern = r'(.+):(\d+): (\w+): (.+) \[(.*)\]'
        match = re.match(pattern, line)
        if not match:
            pattern = r'(.+):(\d+): (\w+): (.+)()'
            match = re.match(pattern, line)

        if match:
            linenumber = match.group(2)
            error_type = match.group(3)
            message = match.group(4)
            sub_type = match.group(5)
            return {"Line Number": linenumber, "Error Type": error_type, "Message": message, "Category": sub_type}
        else:
            logger.debug(("No match found.", line))

            return {"Line Number": "", "Error Type": "", "Message": "", "Category": ""}


class CythonLinter(Linter):
    def run_checker(self, file: str, additional_args: List[str], program_path: str, proj_path: str) -> str:
        # Construct the cython compilation command
        command = ['python', 'setup.py', 'build_ext', '--inplace']
        logger.debug("Running cython compilation command: %s", command)
        
        # Run compilation command and capture the output
        result: CompletedProcess[str] = subprocess.run(command, 
                                                     capture_output=True, 
                                                     text=True, 
                                                     cwd=os.path.abspath(proj_path))
        
        # Return both stdout and stderr since Cython errors appear in stderr
        return result.stdout + result.stderr

    @staticmethod
    def parse_line(line: str) -> Dict[str, Any]:
        # Pattern for Cython error messages
        patterns = [
            r'(.+?):(\d+):(\d+): (.+)',  # Matches file:line:col: message
            r'Error compiling Cython file:',  # Matches compilation error header
            r'------------------------------------------------------------'  # Matches separator
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match and len(match.groups()) == 4:
                file, line_num, col, message = match.groups()
                return {
                    "Line Number": line_num,
                    "Error Type": "Cython Error",
                    "Message": message,
                    "Category": "compilation"
                }
        
        return {"Line Number": "", "Error Type": "", "Message": "", "Category": ""}