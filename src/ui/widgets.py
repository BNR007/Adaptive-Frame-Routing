"""Small reusable widgets for the UniversalUpscaler desktop UI."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class Switch(QAbstractButton):
    """Compact, keyboard-accessible boolean switch."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(36, 20)
        self._offset = 3.0
        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(110)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:
        return QSize(36, 20)

    def _animate(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(19.0 if checked else 3.0)
        self._animation.start()

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = Property(float, get_offset, set_offset)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QColor("#666B73")
        if self.isChecked():
            track = QColor("#6262C9")
        if not self.isEnabled():
            track.setAlpha(110)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0.5, 1.5, 35, 17), 8.5, 8.5)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QRectF(self._offset, 3.5, 13, 13))
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#6262C9"), 1))
            painter.drawRoundedRect(QRectF(0.5, 0.5, 35, 19), 9.5, 9.5)


class SettingsRow(QWidget):
    def __init__(self, title: str, control: QWidget, description: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(14)

        labels = QWidget()
        labels_layout = QVBoxLayout(labels)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("rowTitle")
        labels_layout.addWidget(title_label)
        if description:
            detail = QLabel(description)
            detail.setObjectName("rowDescription")
            detail.setWordWrap(True)
            labels_layout.addWidget(detail)

        control.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(labels, 1)
        layout.addWidget(control, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)


class SettingsCard(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsCard")
        # Cards sharing a grid row may contain very different numbers of rows.
        # A vertical Maximum policy lets the shorter card constrain that grid
        # row, causing Qt to compress and overlap controls in the taller card.
        # Fixed uses the card layout's live size hint, including visible rows.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(16, 14, 16, 14)
        self.content_layout.setSpacing(1)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        self.content_layout.addWidget(heading)
        self.content_layout.addSpacing(6)

    def add_row(self, row: SettingsRow) -> SettingsRow:
        self.content_layout.addWidget(row)
        return row

    def add_widget(self, widget: QWidget) -> QWidget:
        self.content_layout.addWidget(widget)
        return widget

    def add_divider(self) -> None:
        divider = QFrame()
        divider.setObjectName("divider")
        self.content_layout.addWidget(divider)


class SidebarItem(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("sidebarItem")
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class CompactStatusRow(QWidget):
    def __init__(self, title: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        label = QLabel(title)
        label.setObjectName("statusKey")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statusValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)

    def value(self) -> str:
        return self.value_label.text()


class AccentButton(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("accentButton")


class ModeBinding(QWidget):
    """Compatibility adapter exposing the former mode-control interface."""

    value_changed = Signal(str)

    def __init__(self, enabled_switch: Switch, type_widget, *, off: str, parent=None) -> None:
        super().__init__(parent)
        self.hide()
        self._switch = enabled_switch
        self._type_widget = type_widget
        self._off = off
        enabled_switch.toggled.connect(lambda _checked: self.value_changed.emit(self.value()))
        type_widget.currentIndexChanged.connect(lambda _index: self.value_changed.emit(self.value()))

    def value(self) -> str:
        if not self._switch.isChecked():
            return self._off
        return str(self._type_widget.currentData())

    def set_value(self, value: str) -> None:
        previous = self.value()
        self._switch.blockSignals(True)
        self._type_widget.blockSignals(True)
        self._switch.setChecked(value != self._off)
        self._switch.set_offset(19.0 if value != self._off else 3.0)
        index = self._type_widget.findData(value)
        if index >= 0:
            self._type_widget.setCurrentIndex(index)
        self._type_widget.blockSignals(False)
        self._switch.blockSignals(False)
        if self.value() != previous:
            self.value_changed.emit(self.value())


# Compatibility names retained for downstream imports.
OptionRow = SettingsRow
SectionCard = SettingsCard
