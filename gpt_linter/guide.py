import os
import anthropic
from guidance import models, gen, user, system, assistant
import guidance


import logging
import os
from guidance import models, gen, user, system, assistant
import guidance
import ratelimit
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

@on_exception(expo, RateLimitException, max_tries=4)
@ratelimit.limits(calls=10, period=60)
def mygen(*args, **kwargs):
    logging.debug(f"mygen called with args: {args}, kwargs: {kwargs}")
    if 'rate_limit' in kwargs:
        kwargs.pop('rate_limit')
    response = gen(*args, **kwargs)
    logging.debug(f"mygen response: {response}")
    return response

class Guidance:
    llm = None
    @staticmethod
    def set_key(args):
        if args.provider == 'anthropic':
            if 'ANTHROPIC_API_KEY' not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            Guidance.llm = models.Anthropic(args.model, api_key=os.environ['ANTHROPIC_API_KEY'], echo=False)
        else:
            if 'OPEN_AI_KEY' not in os.environ:
                raise ValueError("OPEN_AI_KEY environment variable not set")
            Guidance.llm = models.OpenAI(args.model, api_key=os.environ['OPEN_AI_KEY'], echo=False)

    @classmethod
    @on_exception(expo, RateLimitException, max_tries=10)
    @ratelimit.limits(calls=10, period=60)
    def safe_concat(cls, addition):
        logging.debug(f"safe_concat called with addition: {addition}")
        try:
            result = cls.llm + addition
            if result is None:
                raise RateLimitException("Rate limit exceeded", period_remaining=60)
            cls.llm = result
            logging.debug(f"safe_concat successful, new llm state: {result}")
        except anthropic.RateLimitError as e:
            logging.error(f"An error occurred: {str(e)}")
            raise RateLimitException(f"An error occurred: {str(e)}", period_remaining=60)

    @classmethod
    def guide_for_errors(cls, args):
        def error_handler(filename, file, error):
            with user():
                cls.safe_concat(f'''What is the fix for this issue on {filename}?
                {error}
                Be sure to inspect the entire relevant context in the file before suggesting a fix.
                Be short and precise regarding the fix, and refer to the change in code. 
                In your answer, you shouldn't repeat instructions given to you, and you shouldn't include many lines of code.
                If you dont know the answer, just say NOFIX.''')
            
            with assistant():
                cls.safe_concat(mygen('fix', list_append=True, temperature=args.temperature_per_fix, max_tokens=args.max_tokens_per_fix))

        def create_program(filename, file, errors, reference_file=None):
            with system():
                cls.safe_concat('''You are a helpful assistant. You will be given a file and an issue. 
                You need to come up with fixes for the issue, even if it is a minor issue.''')

            reference_text = f"\nHere is a reference file that might help:\n{reference_file}" if reference_file else ""
            with user():
                cls.safe_concat(f'''Given this {filename}: 
                {file}{reference_text}
                A list of issues will be given''')

            for error in errors:
                error_handler(filename, file, error)
                import time
                time.sleep(2)

            return cls.llm
        return create_program

    @classmethod
    def guide_for_fixes(cls, args):
        def create_program(file, fixes):
            with system():
                cls.safe_concat('''You are a helpful assistant. You will be given a list of corrections to do in a file, 
                and will update the file accordingly.
                Reply only with xml that has the following format:
                ```xml
                <file>
                    <codsection startline="STARTLINE" endline="ENDLINE">
                        <oldlines>original code that will be replaced</oldlines>
                        <newlines>new code that replaces the old code</newlines>
                    </codsection>
                    <codsection startline="line2" endline="endline2">
                        <oldlines>original code section 2</oldlines>
                        <newlines>new code section 2</newlines>
                    </codsection>
                    ...
                </file>                                            
                ```
                
                ''')

            with user():
                cls.safe_concat(f'''This is the file:
                {file}
                Those are the fixes:
                {chr(10).join(f"- {fix}" for fix in fixes)}
                Make sure you apply all the corrections in the resulted file. 
                You should take note of all the relevant places you need to fix.

                leave a  comment if you cant solve it.''')

            with assistant():
                cls.safe_concat(mygen('fixedfile', temperature=args.temperature_for_file, max_tokens=args.max_tokens_for_file))
            return cls.llm

        return create_program
