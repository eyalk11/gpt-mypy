import os
import xml

import logging
import xml.etree.ElementTree as ET

import ratelimit

from gpt_linter.common import generate_diff
from gpt_linter.guide import Guidance
from gpt_linter.linter import MyPyLinter, CythonLinter

from gpt_linter.logger import Logger
logger=Logger()


MYPYARGS = ['--disallow-untyped-defs']

DEFAULT_TOKENS =100000
DEFAULT_TOKENS_PER_FIX =8000
DEFAULT_TEMP_PER_FIX=0.7
DEFAULT_TEMP = 0.2
import argparse
from typing import List, Dict, Any, Optional, Iterator
import subprocess

from gpt_linter.simpleexceptioncontext import simple_exception_handling, SimpleExceptionContext


# Add at the top with other constants
DEFAULT_PROVIDER = "anthropic"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_MODEL = "gpt-3.5-turbo-16k"

# Add these constants at the top with other defaults
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8000

# Add this with other global constants at the top of the file

CHUNK_SIZE = 30  # Number of errors to process in each batch

# Add at the top with other constants
FILE_CHUNK_SIZE = 6000  # Number of lines to process in each file chunk

class GPTLinter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.file_name = args.file
        self.current_content = open(self.file_name, 'rt').read()
        self.total_lines = len(self.current_content.split('\n'))#waste
        self.current_content = self.current_content
        self.debug = args.debug
        
        # Choose appropriate linter based on file extension
        if self.file_name.endswith('.pyx') or self.file_name.endswith('.pxd'):
            self.linter = CythonLinter()
        else:
            self.linter = MyPyLinter()
        self.lines = self.current_content.split('\n')
    @staticmethod
    def escape_marked_sections(content: str) -> str:
            """Escape XML content between ### markers."""
            import re
            from xml.sax.saxutils import escape
            
            def escape_match(match):
                inner = match.group(1)
                return f"{escape(inner)}"
                
            pattern = r">[\n\t]*###(.*?)###[\n\t]*<"
            return re.sub(pattern, escape_match, content, flags=re.DOTALL)

    @simple_exception_handling(err_description="Failed to get new content", return_succ=None)
    def get_new_content(self, fixes: Dict[str, Any], issues: List[str]) -> Optional[str]:
        fix_guide= Guidance.guide_for_fixes(self.args)
        fix_res=fix_guide(file=self.current_content, issues_and_fixes=zip(issues,fixes) )
        lines = self.current_content.split('\n')
        if not 'fixedfile' in fix_res:
            logger.error('no fixed file')
            return None
        fixed = fix_res['fixedfile']
        logger.debug(f'fixed file: {fixed}')
        fixed= self.escape_marked_sections(fixed)
        logger.debug(f'fixed file escaped: {fixed}')
        #bb = json.loads(fix_res["fixedfile"])['file']
        for k in range(3):

            if k==1:
                try:
                    logger.debug("trying to fix bad formatting")
                    new_content=fixed[fixed.index('<file>'):]
                    new_content=new_content[:new_content.rindex('</file>')+len('</file>')]
                except Exception as e :
                    logger.error(e)
                    logger.error(f"bad formatting manual extraction {k}")
                    continue
            elif k==0:
                try:
                    new_content= fixed[fixed.index('```xml') + 6:]
                    new_content=new_content[:new_content.rindex('```')]
                except:
                    logger.debug("no ```xml found ")
                    if len(fixed)>100:
                        new_content=fixed
                    else:
                        return None
            elif k==2:
                try:
                    new_content= fixed[fixed.index('```python') + 9:] #it wasn't supposed to be like that
                    new_content=new_content[:new_content.rindex('```')]
                except:
                    logger.debug("no ```python found ")

            try:
                root=ET.fromstring(new_content) #remove the file element
                new_content=self.get_new_content_int(root)
                return new_content

            except xml.etree.ElementTree.ParseError:
                logger.debug(f"bad formatting {k}")
                if k == 2:
                    raise
                # logger.debug(new_content)
                # if new_content.startswith('<file>') and new_content.endswith('</file>') and len(new_content) > 0.8 * len(self.current_content):
                #     new_content= new_content[len('<file>'):(-1)*len('</file>')] #it is probably too idiot to extract it.
                #     return new_content
                # if k == 2 and len(new_content) >100:
                #     logger.warn("will try anyway")
                #     return new_content

        logger.debug("got to the end")
        return None

    @simple_exception_handling(err_description="Failed to get new content internal", return_succ=None)
    def get_new_content_int(self, xml_content) -> Optional[str]:
        try:
            sections = self.parse_code_sections(xml_content)
            
            if not sections:
                logger.error('No valid code sections found')
                return None
                
            # Convert file content to lines
            lines = self.current_content.split('\n')
            
            # Process each section
            for section in sections:
                start, end = section['start'] - 1, section['end'] - 1  # Convert to 0-based indexing
                old_code = ''.join(c for c in section['old_code'] if c.isalnum() or c.isspace())
                new_lines = section['new_code'].split('\n')
                
                # Find actual location using old_code
                actual_start = start
                window_size = end - start + 1
                min_distance = float('inf')
                
                # Search in a reasonable range around the suggested line numbers
                search_range = 20  # Adjust this value as needed
                search_start = max(0, start - search_range)
                search_end = min(len(self.lines), end + search_range + 1)
                
                for i in range(search_start, search_end - window_size + 1):
                    window = ''.join(c for c in '\n'.join(self.lines[i:i + window_size]) if c.isalnum() or c.isspace())
                    if window == old_code:
                        # Calculate distance from suggested position
                        distance = abs(i - start)
                        if distance < min_distance:
                            min_distance = distance
                            actual_start = i
                
                # Replace the lines at the found position
                self.lines[actual_start:actual_start + window_size] = new_lines
                
            return '\n'.join(self.lines)
            
        except Exception as e:
            logger.error(f'Failed to process content: {e}')
            return None

    @simple_exception_handling(err_description="Failed to get issues string", return_succ=[])
    def get_issues_string(self, issues: List[Dict[str, Any]]) -> Iterator[str]:
        lines=self.current_content.split('\n')

        for issue in issues:
            try:
                ln=int(issue['Line Number'])
                line_range= '\n'.join( lines[ max(ln-1,0) :min(ln+1,len(lines))])

                issue[f"lines {ln-1} to {ln+1} in the file"]='\n'+line_range

                st='\n'.join(f"{k}: {v}" for k,v in issue.items())

                logger.debug(st)
                yield st
            except:
                logger.debug(issue)
                continue

    def main(self) -> None:
        logger.setup_logger(self.debug)

        if 'OPEN_AI_KEY' not in os.environ:
            logger.error('OPEN_AI_KEY not set')
            return
        Guidance.set_key(self.args)
        
        # Get all errors once
        logger.info("mypy output:")
        errors = self.linter.get_issues(self.args)
        logger.debug(errors)
        
        if not errors or (isinstance(errors, list) and len(errors) == 0):
            logger.info('No type checking errors found')
            return

        # Process file in chunks if it's too large
        if self.total_lines > FILE_CHUNK_SIZE:
            self.process_large_file(errors)
        else:
            self.try_to_solve_issues(errors)

    def process_large_file(self, all_errors: List[Dict[str, Any]]) -> None:
        """Process a large file in chunks."""
        num_chunks = (self.total_lines + FILE_CHUNK_SIZE - 1) // FILE_CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            start_line = chunk_idx * FILE_CHUNK_SIZE
            end_line = min((chunk_idx + 1) * FILE_CHUNK_SIZE, self.total_lines)
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (lines {start_line + 1}-{end_line})")
            
            # Store original content
            original_content = self.current_content
            
            # Set current content to chunk
            self.current_content = '\n'.join(self.lines[start_line:end_line])
            if start_line>0:
                self.current_content="...\n" + self.current_content
            if end_line<len(self.lines):
                self.current_content=self.current_content + "\n..."
            # Filter errors for current chunk
            chunk_errors = [
                error for error in all_errors 
                if start_line < int(error.get('Line Number', 0)) <= end_line
            ]
            
            if chunk_errors:
                # Process the chunk
                self.try_to_solve_issues(chunk_errors)
                
                # If changes were made, update the lines array
                if self.current_content != '\n'.join(self.lines[start_line:end_line]):
                    new_lines = self.current_content.split('\n')
                    self.lines[start_line:end_line] = new_lines
                    
                # Generate diff file for this chunk if requested
                if self.args.diff_file:
                    chunk_diff_file = f"{self.args.diff_file}.chunk{chunk_idx + 1}"
                    colored_diff, diff = generate_diff(
                        '\n'.join(self.lines[start_line:end_line]),
                        self.current_content,
                        f"{self.args.file}:lines_{start_line + 1}-{end_line}"
                    )
                    with open(chunk_diff_file, 'wt') as f:
                        f.write(diff)
                    
            # Restore full content for next iteration
            self.current_content = original_content
            
        # Update the final content
        self.current_content = '\n'.join(self.lines)
        
        # Write the final result
        if not self.args.dont_ask:
            print("Apply all changes? (y/n)")
            if input().lower() == 'y':
                with open(self.args.file, 'wt') as f:
                    f.write(self.current_content)

    def try_to_solve_issues(self, errors):
        logger.info("trying to solve issues")
        
        # Process errors in chunks
        total_errors = len(errors)
        for i in range(0, total_errors, CHUNK_SIZE):
            chunk = errors[i:i + CHUNK_SIZE]
            logger.info(f"Processing chunk {i//CHUNK_SIZE + 1} of {(total_errors + CHUNK_SIZE - 1)//CHUNK_SIZE}")
            
            remaining_errors = self.process_error_chunk(chunk, errors)
            if remaining_errors is None:  # User quit or error occurred
                return
            errors = remaining_errors
            
            if len(errors) == 0:
                return
            
            if i + CHUNK_SIZE < total_errors:
                print("Continue with next chunk? (y/n)")
                if input().lower() != 'y':
                    return
                
    @simple_exception_handling(err_description="Failed to process error chunk", return_succ=None)
    def process_error_chunk(self, chunk, all_errors):
        """Process a chunk of errors and return remaining errors or None if processing should stop."""
        errors=list(self.get_issues_string(chunk)) 
        err_res = self.get_fixes(errors)
        new_content = self.get_new_content(err_res,errors)
        
        if new_content is None:
            logger.error('cant continue')
            return None
        
        colored_diff, diff = generate_diff(self.current_content, new_content, self.args.file.replace("\\", '/'))
        
        if self.args.diff_file:
            with open(self.args.diff_file, 'wt') as f:
                f.write(diff)
        
        old_errors = all_errors
        if not self.args.recheck_policy == 'none':
            errors = self.check_new_file(new_content)
        
        print(diff if self.args.no_color else colored_diff)
        update = False
        
        if (len(errors) == 0 and self.args.auto_update == 'strict') \
            or (self.args.auto_update == 'permissive' and len(old_errors) > len(errors)):
            update = True
        elif not self.args.dont_ask:
            print("Press 'w' to write and quit. Press 'y' to write and continue testing, and 'q' to quit without writing.")
            choice = input().lower()
            if choice == 'y':
                update = True
            elif choice == 'w':
                update = True
                self.args.recheck_policy = 'none'
            elif choice == 'q':
                return None
        
        if update:
            if self.args.stash:
                if not self.stash_changes(self.args.file):  # Changed to self.stash_changes
                    if not self.args.dont_ask:
                        print("Failed to stash changes. Continue anyway? (y/n)")
                        if input().lower() != 'y':
                            return None
            
            with open(self.args.file, 'wt') as f:
                f.write(new_content)
            self.current_content = new_content
        
        return errors

    @simple_exception_handling(err_description="Failed to check new file", return_succ=[])
    def check_new_file(self, new_content: str) -> List[Dict[str, Any]]:
        newfile: str = self.args.file.replace('.py', '.fixed.py')  # must be in the same folder sadly.
        open(newfile, 'wt').write(new_content)
        logger.info('output from mypy after applying the fixes:')
        try:
            return self.linter.get_issues(self.args, override_file=newfile)
        finally:
            if not self.args.store_fixed_file:
                try:
                    os.remove(newfile)
                except:
                    logger.error('could not remove file %s' % newfile)


    @simple_exception_handling(err_description="Failed to get fixes", return_succ={"fix": []})
    def get_fixes(self, errors: List[str]) -> Dict[str, Any]:
        err_guide = Guidance.guide_for_errors(self.args)
        reference = self.get_reference_content()
        err_res: Dict[str, Any] = err_guide(filename=self.file_name, 
                                    file=self.current_content, 
                                    errors=errors,
                                    reference_file=reference)
        if not self.args.dont_print_fixes:
            logger.info('suggested fixes:')
            # Filter out NOFIX responses and log them with context
            fixes = []
            for fix, error in zip(err_res['fix'], errors):
                if fix.strip() == 'NOFIX':
                    logger.warning(f'No fix available for error:\n{error}\n---')
                else:
                    fixes.append(fix)
            
            logger.info('\n'.join(fixes))
        return fixes


    @simple_exception_handling(err_description="Failed to get reference content", return_succ="")
    def get_reference_content(self) -> str:
        if not self.args.ref_file or not os.path.exists(self.args.ref_file):
            return ""
        
        with open(self.args.ref_file, 'rt') as f:
            return f.read()

    @staticmethod
    @simple_exception_handling(err_description="Failed to parse code sections", return_succ=[])
    def parse_code_sections(xml_content: str) -> List[Dict[str, Any]]:
        try:
            sections = []
            for section in xml_content.findall('codsection'):
                start = int(section.get('startline'))
                end = int(section.get('endline'))
                old_lines = section.find('oldlines').text.strip() if section.find('oldlines') is not None else ""
                new_lines = section.find('newlines').text.strip() if section.find('newlines') is not None else ""
                sections.append({
                    'start': start,
                    'end': end,
                    'old_code': old_lines,
                    'new_code': new_lines
                })
            return sections
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return []
    @staticmethod
    @simple_exception_handling(err_description="Failed to stash changes", return_succ=False)
    def stash_changes(file_path: str) -> bool:
        """
        Stash changes for the specified file before applying fixes.
        Returns True if stashing was successful.
        """
        try:
            # Add file to git index if not already tracked
            subprocess.run(['git', 'add', file_path], check=True)
            # Stash changes with a descriptive message
            stash_msg = f"GPT-Linter: Stashing changes for {file_path}"
            subprocess.run(['git', 'stash', 'push', '-m', stash_msg], check=True)
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to stash changes. Make sure you're in a git repository.")
            return False

    
def main() -> None:
    # Create the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Run type checking on Python/Cython files and use OpenAI GPT to fix the errors.')
    
    # Add existing arguments
    parser.add_argument('file', help='Python or Cython file to check')
    parser.add_argument('--proj-path', '-p', default='.', 
                       help='Path to project (required for Cython compilation)')
    
    # Add Cython-specific arguments
    parser.add_argument('--cython-args', nargs=argparse.REMAINDER,
                       help='Additional options for Cython compilation')

    # Add the arguments
    parser.add_argument('mypy_args', nargs=argparse.REMAINDER, help=f'Additional options for mypy after --. By default, uses {MYPYARGS}')
    parser.add_argument('--mypy-path', default='mypy', help='Path to mypy executable (default: "mypy")')
    parser.add_argument('--error_categories', action='store', help='Type of errors to process')
    parser.add_argument('--max_errors', action='store', type=int, default=10, help='Max number of errors to process per cycle')
    parser.add_argument('-d','--diff-file', action='store', help='Store diff in diff file')
    parser.add_argument('-s','--store-fixed-file', action='store_true', default=False, help='Keeps file.fixed.py')

    parser.add_argument('--dont-ask', action='store_true', default=False,
                        help='Dont ask if to apply to changes. Useful for generting diff')
    parser.add_argument('-m','--model', default=None, help='Openai model to use')
    parser.add_argument('--max_tokens-per-fix', default=DEFAULT_TOKENS_PER_FIX, help='tokens to use for generating each fix')
    parser.add_argument('--temperature-per-fix', default=DEFAULT_TEMP_PER_FIX, help='temperature to use for fixes')
    parser.add_argument('--max_tokens_for_file', default=DEFAULT_TOKENS, help='tokens to use for file')
    parser.add_argument('--temperature_for_file', default=DEFAULT_TEMP, help='temperature to use for generating the file')
    parser.add_argument('-r','--recheck-policy', choices=['recheck','none','recheckandloop'],  default='recheckandloop',
                        help='Recheck the file for issues before suggesting a fix. require to temporarily save file.fixed.py (has to be in the project). recheckandloop will go for another loop if done fixing and there are still errors.')

    parser.add_argument('--debug', action='store_true', default=False, help='debug log level ')
    parser.add_argument('-a','--auto-update', choices=['permissive','no','strict'], default='no', help='auto update file if no errors (if strict). On permissive policy it updates if the number of errors decreased. ')
    parser.add_argument('-D','--dont-print-fixes', action='store_true', default=False, help='dont print fixes')
    #add no colors option  
    parser.add_argument('-N','--no_color', action='store_true', help='dont print color diff')

    parser.add_argument('--ref-file', help='Reference file to use for additional context')
    parser.add_argument('--provider', choices=['openai', 'anthropic'], default=DEFAULT_PROVIDER,
                   help='AI provider to use (default: openai)')
   
    parser.add_argument('--stash', action='store_true', help='Stash changes before applying fixes')
    parser.add_argument('--chunk-size', type=int, default=3, help='Number of errors to process in each chunk')
    # Modify the argument parser section
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, 
                       help='Temperature to use for all generations')
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS,
                       help='Max tokens to use for all generations')
    # Parse the arguments
    args: argparse.Namespace = parser.parse_args()
    if args.provider == 'anthropic' and args.model is None:
        args.model = ANTHROPIC_DEFAULT_MODEL
    elif args.provider == 'openai' and args.model is None:
        args.model = DEFAULT_MODEL
    if len(args.mypy_args) ==0:
        args.mypy_args = MYPYARGS

    GPTLinter(args).main()


if __name__ == '__main__':
    main()









