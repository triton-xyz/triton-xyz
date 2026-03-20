#pragma once

#include "Data/TreeData.h"

namespace proton {

class CpuInstrumentationTreeData : public TreeData {
public:
  using TreeData::TreeData;

protected:
  void enterScope(const Scope &scope) override;
  void exitScope(const Scope &scope) override;
};

} // namespace proton
