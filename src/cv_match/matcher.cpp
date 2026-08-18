#include "base_matcher.h"
#include "PatternMatching.h"
#include "ShapeMatching.h"

#include <fstream>

namespace template_matching
{
	auto logger_ = spdlog::stdout_color_mt("template_matching");  //spdlog::rotating_logger_mt<spdlog::async_factory>("file_logger", "run.log", 1024 * 1024 * 5, 1);

	std::unique_ptr<Matcher> GetMatcher(const MatcherParam& param)
	{
		MatcherParam paramCopy = param;
		std::unique_ptr<BaseMatcher> matcher;
		
		switch (paramCopy.matcherType)
		{
		case MatcherType::PATTERN:
			logger_->info("Initializing matcher for type: PATTERN");
			matcher = std::make_unique<PatternMatcher>(paramCopy);
			break;
		case MatcherType::SHAPE:
			logger_->info("Initializing matcher for type: SHAPE");
			matcher = std::make_unique<ShapeMatcher>(paramCopy);
			break;
		default:
			logger_->error("Unknown matcherType");
			return nullptr;
		}

		if (!matcher || !matcher->isInited())
		{
			return nullptr;
		}

		return matcher;
	}
	
}